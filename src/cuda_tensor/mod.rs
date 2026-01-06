//! GPU-backed tensor implementation and CUDA helpers. Only available with `--features=cuda`
//!
//! This module provides a `GpuTensor<T>` implementation of the crate `Tensor`
//! trait which stores data in CUDA device memory and exposes common
//! tensor operations (element-wise ops, matrix multiply, reduction, etc.)
//! implemented using CUDA kernels and cuBLAS where appropriate.
//!
//! Important notes for production use:
//! - The global `GPU_CONTEXT` must be initialized before creating or using
//!   any `GpuTensor` instances.
//! - Many helpers launch device kernels and thus rely on correct stream and
//!   kernel linkage; errors from kernel launches are surfaced as `Err(String)`
//!   in fallible APIs.
//! - Only tensors with up to 2 dimensions are currently supported for some
//!   operations (e.g. transpose) — callers should validate shape expectations.
use crate::cuda_tensor::custom_device_buffer::{
    get_device_buffer, get_device_buffer_from_slice, CustomDeviceBuffer,
};
use crate::numeric::FloatingPoint;
use crate::numeric::{Numeric, SignedNumeric};
use crate::tensor::math::TensorMath;
use crate::GPU_CONTEXT;
use core::ffi::c_void;
use cublas_sys::*;
use cust::memory::bytemuck::Zeroable;
use std::ops::{Add, Mul, Neg, Sub};

use crate::Tensor;
mod custom_device_buffer;
/// A lightweight device memory pool to reduce allocation overhead.
mod mem_pool;
/// Public re-export of the CUDA memory pool manager.
pub use mem_pool::CudaMemoryPool;
mod cublas_handle;
/// Public re-export for managing cuBLAS handles scoped to the GPU context.
pub use cublas_handle::CublasHandle;
use cust::launch;
use cust::memory::DeviceCopy;
use cust::prelude::DeviceBuffer;
use cust::prelude::Function;
use cust::stream::Stream;

/// Operation selector for element-wise device kernels.
///
/// This enum is passed to device kernels to select the requested element-wise
/// operation (scale, exp, trig functions, activation functions, etc.). The
/// numeric discriminants are matched in the CUDA kernels.
#[derive(Clone, Copy, Debug)]
enum OpType {
    Exp = 0,
    Sin = 2,
    Cos = 3,
    Tan = 4,
    Tanh = 5,
    Sigmoid = 6,
    Log = 7,
    Ln = 8,
    GreaterThanZeroMask = 9,
    ReLU = 10,
}

/// Selector for binary element-wise arithmetic kernels.
///
/// Used by the vector arithmetic kernel to select addition/subtraction/
/// multiplication/division between two tensors.
#[derive(Clone, Copy, Debug)]
enum ArithmaticType {
    Add = 1,
    Sub = 2,
    Mul = 3,
    Div = 4,
}

/// A tensor whose backing storage resides on a CUDA device.
///
/// `GpuTensor<T>` stores its shape as a `Vec<u32>` and owns a
/// `CustomDeviceBuffer<T>` that contains the device memory. The generic `T`
/// must implement `Numeric` and `DeviceCopy` so it can be moved between host
/// and device memory. Most public operations return `Result<Self, String>` to
/// surface errors coming from invalid shapes or CUDA/kernel failures.
#[derive(Debug)]
pub struct GpuTensor<T: Numeric + DeviceCopy> {
    shape: Vec<u32>,
    device_buffer: CustomDeviceBuffer<T>,
}

impl<T: Numeric + Zeroable + DeviceCopy> Tensor<T> for GpuTensor<T> {
    /// Create a new `GpuTensor` from host `data` and move it to device memory.
    ///
    /// Returns `Err(String)` when the provided data length does not match the
    /// product of `shape` or if other allocation errors occur.
    fn new(shape: Vec<u32>, data: Vec<T>) -> Result<Self, String> {
        Self::_new(shape, data)
    }

    /// Synchronously copy the tensor contents from device to host and return
    /// them as a `Vec<T>`.
    fn get_data(&self) -> Vec<T> {
        self._data()
    }

    /// Return the tensor shape as a slice of dimensions.
    fn get_shape(&self) -> &Vec<u32> {
        &self.shape
    }

    /// Element-wise addition of two tensors. Returns an error on shape
    /// mismatch.
    fn add(&self, rhs: &Self) -> Result<Self, String> {
        self._element_arithmatic(rhs, ArithmaticType::Add)
    }

    /// Element-wise subtraction. Returns an error on shape mismatch.
    fn sub(&self, rhs: &Self) -> Result<Self, String> {
        self._element_arithmatic(rhs, ArithmaticType::Sub)
    }

    /// Element-wise multiplication (Hadamard product). Returns an error on
    /// shape mismatch.
    fn matmul(&self, rhs: &Self) -> Result<Self, String> {
        self._mul(rhs)
    }

    /// Element-wise division. Returns an error on shape mismatch.
    fn div(&self, rhs: &Self) -> Result<Self, String> {
        self._element_arithmatic(rhs, ArithmaticType::Div)
    }

    /// Matrix multiplication. Validates dimensions and performs GPU-backed
    /// multiply (cuBLAS is used when available).
    fn mul(&self, rhs: &Self) -> Result<Self, String> {
        self._element_arithmatic(rhs, ArithmaticType::Mul)
    }

    /// Transpose the tensor (only supported up to 2D). Returns a new
    /// `GpuTensor` containing the transposed result.
    fn t(&self) -> Result<Self, String> {
        self._t()
    }

    /// Scale each element by `scalar`.
    fn scale(&self, scalar: T) -> Result<Self, String> {
        self._s(scalar)
    }

    /// Synchronize the current CUDA stream and block until all device work
    /// submitted to the global GPU context has completed.
    fn synchronize() {
        let _ = &(GPU_CONTEXT
            .get()
            .expect("No GPU Context Intialized")
            .stream
            .as_ref()
            .expect("Stream could not be found"))
        .synchronize();
    }

    /// Create a tensor of zeros with the given `shape`.
    fn zeroes(shape: &[u32]) -> Self {
        Self::_new_with_value(shape.to_vec(), T::zero()).unwrap()
    }

    /// Create a tensor filled with ones with the given `shape`.
    fn ones(shape: &[u32]) -> Self {
        Self::_new_with_value(shape.to_vec(), T::one()).unwrap()
    }

    /// Clip tensor values to the closed interval [`min`, `max`].
    fn clip(&self, min: T, max: T) -> Result<Self, String> {
        self._clip(min, max)
    }

    /// Sum all elements of the tensor and return a 1-element tensor
    /// containing the total.
    fn sum(&self) -> Result<Self, String> {
        let data = self._column_sum().unwrap().get_data();

        let total: T = data.iter().fold(T::zero(), |acc, &x| acc + x);

        Self::new(vec![1], vec![total])
    }
}

impl<T: Numeric + Zeroable + DeviceCopy> Add for GpuTensor<T> {
    type Output = Result<Self, String>;

    fn add(self, rhs: Self) -> Result<Self, String> {
        self._element_arithmatic(&rhs, ArithmaticType::Add)
    }
}

impl<T: Numeric + Zeroable + DeviceCopy> Sub for GpuTensor<T> {
    type Output = Result<Self, String>;

    fn sub(self, rhs: Self) -> Result<Self, String> {
        self._element_arithmatic(&rhs, ArithmaticType::Sub)
    }
}

impl<T: Numeric + Zeroable + DeviceCopy> Mul for GpuTensor<T> {
    type Output = Result<Self, String>;

    fn mul(self, rhs: Self) -> Result<Self, String> {
        self._mul(&rhs)
    }
}

impl<T: SignedNumeric + Zeroable + DeviceCopy> Neg for GpuTensor<T> {
    type Output = Result<Self, String>;
    fn neg(self) -> Result<Self, String> {
        self._s(-T::one())
    }
}

impl<T: Numeric + Zeroable + DeviceCopy> PartialEq for GpuTensor<T> {
    fn eq(&self, other: &Self) -> bool {
        println!("Partial eq called");
        self._eq(other)
    }
}

impl<T: Numeric + Zeroable + DeviceCopy> GpuTensor<T> {
    fn _get_function(fn_name: &str) -> Function<'_> {
        let t = GPU_CONTEXT
            .get()
            .expect("No GPU Context Set")
            .get_function(fn_name);
        t
    }

    fn _get_stream() -> &'static Stream {
        GPU_CONTEXT
            .get()
            .expect("No GPU Context Set")
            .stream
            .as_ref()
            .expect("Stream not found")
    }

    fn _get_cublas_handle() -> cublasHandle_t {
        GPU_CONTEXT
            .get()
            .expect("No GPU Context Set")
            .cublas_handle
            .handle
    }

    fn _eq(&self, other: &Self) -> bool {
        if self.shape != other.shape {
            return false;
        }

        let total_size = self.shape.iter().product::<u32>() as usize;

        if total_size == 0 {
            return true;
        }

        for i in 0..total_size {
            let diff = f32::abs(self.get_data()[i].f32() - other.get_data()[i].f32());

            if diff > f32::EPSILON * 10.0f32 {
                return false;
            }
        }
        true
    }

    fn element_op(&self, op_type: OpType) -> Result<Self, String> {
        let total_elements: u32 = self.shape.iter().product();
        let threads_per_block = 1024; // Use a typical 1D block size
        let grid_1d = total_elements.div_ceil(threads_per_block);

        let result = self._get_initialized_buffer(total_elements as usize);

        let stream = Self::_get_stream();

        let operation = Self::_get_function("element_op");

        unsafe {
            let _ = launch!(operation<<<grid_1d, threads_per_block, 0, stream>>>(
                self.device_buffer.as_device_ptr(),
                result.as_device_ptr(),
                total_elements as i32,
                op_type as i32,
                1.0_f32
            ));
        };

        Ok(Self::_with_device_buffer(self.shape.clone(), result))
    }

    fn _s(&self, scale: T) -> Result<Self, String> {
        let total_elements: u32 = self.shape.iter().product();
        let threads_per_block = 1024; // Use a typical 1D block size
        let grid_1d = total_elements.div_ceil(threads_per_block);

        let result = self._get_initialized_buffer(total_elements as usize);

        let stream = Self::_get_stream();

        let operation = Self::_get_function("element_op");

        unsafe {
            let _ = launch!(operation<<<grid_1d, threads_per_block, 0, stream>>>(
                self.device_buffer.as_device_ptr(),
                result.as_device_ptr(),
                total_elements as i32,
                1_i32,
                scale
            ));
        };

        Ok(Self::_with_device_buffer(self.shape.clone(), result))
    }

    fn _clip(&self, min: T, max: T) -> Result<Self, String> {
        let total_elements: u32 = self.shape.iter().product();
        let threads_per_block = 1024; // Use a typical 1D block size
        let grid_1d = total_elements.div_ceil(threads_per_block);

        let result = self._get_initialized_buffer(total_elements as usize);

        let stream = Self::_get_stream();

        let operation = Self::_get_function("clip");

        unsafe {
            let _ = launch!(operation<<<grid_1d, threads_per_block, 0, stream>>>(
                self.device_buffer.as_device_ptr(),
                result.as_device_ptr(),
                total_elements as i32,
                min,
                max
            ));
        };

        Ok(Self::_with_device_buffer(self.shape.clone(), result))
    }

    fn _t(&self) -> Result<Self, String> {
        if self.shape.len() > 2 {
            return Err("Only upto 2D tensors can be transposed.".to_string());
        }

        if self.shape.len() == 1 {
            unsafe {
                let new_device_buffer = CustomDeviceBuffer {
                    device_buffer: DeviceBuffer::from_raw_parts(
                        self.device_buffer.as_device_ptr(),
                        self.shape[0] as usize,
                    ),
                };

                return Ok(Self::_with_device_buffer(
                    self.shape.clone(),
                    new_device_buffer,
                ));
            }
        }

        let transpose = Self::_get_function("transpose_naive");

        let result = self._get_initialized_buffer(self.shape.iter().product::<u32>() as usize);

        let new_shape = vec![self.shape[1], self.shape[0]];

        const BLOCK_DIM_X: u32 = 16;
        const BLOCK_DIM_Y: u32 = 16;

        let m = self.shape[0] as i32;
        let n = self.shape[1] as i32;

        let grid_x = (n as u32).div_ceil(BLOCK_DIM_X);
        let grid_y = (m as u32).div_ceil(BLOCK_DIM_Y);

        let stream = Self::_get_stream();

        unsafe {
            let _ = launch!(transpose<<<(grid_x, grid_y, 1), (BLOCK_DIM_X, BLOCK_DIM_Y, 1), 0, stream>>>(
                self.device_buffer.as_device_ptr(),
                result.as_device_ptr(),
                m,
                n
                )

            );
        }
        Ok(Self::_with_device_buffer(new_shape, result))
    }

    fn _data(&self) -> Vec<T> {
        let total_elements = self.shape.iter().product::<u32>() as usize;

        let mut data = vec![T::zero(); total_elements];

        unsafe {
            cust::sys::cuMemcpyDtoH_v2(
                data.as_mut_ptr() as *mut c_void,
                self.device_buffer.as_device_ptr().as_raw(),
                total_elements * size_of::<T>(),
            )
        };
        data
    }

    fn _shape(&self) -> Vec<u32> {
        self.shape.clone()
    }

    fn _element_arithmatic(&self, rhs: &Self, op: ArithmaticType) -> Result<Self, String> {
        if self.shape != rhs.shape {
            return Err(format!("ShapeMismatch:The dimensions of two matrices are not compatible for addition/subtraction- {:?} {:?}", self.shape, rhs.shape));
        }

        let add = Self::_get_function("vector_arithmatic");

        let total_elements = self.shape.iter().product::<u32>() as usize;

        let result = self._get_initialized_buffer(total_elements);

        let total_size_u32 = total_elements as u32;
        let threads_per_block = 1024;

        let grid_1d = total_size_u32.div_ceil(threads_per_block);

        let stream = Self::_get_stream();
        unsafe {
            let _ = launch!(add<<< grid_1d, threads_per_block, 0, stream >>>(
                self.device_buffer.as_device_ptr(),
                rhs.device_buffer.as_device_ptr(),
                result.as_device_ptr(),
                total_size_u32 as i32,
                op as u32
            ));
        }

        Ok(Self::_with_device_buffer(self.shape.clone(), result))
    }

    fn _column_sum(&self) -> Result<Self, String> {
        // Assuming self.shape is [rows, cols]
        let num_rows = self.shape[0];

        let num_cols = match self.shape.len() {
            2 => self.shape[1],
            _ => 1,
        };

        // 1. Get the kernel function
        let sum_kernel = Self::_get_function("column_reduce");

        // 2. Prepare result buffer: Only need space for 'num_cols' floats
        let result = self._get_initialized_buffer(num_cols as usize);

        // 3. Grid Setup: One thread per column
        // Your kernel uses: int col = blockIdx.x * blockDim.x + threadIdx.x;
        let threads_per_block = 256;
        let grid_size = num_cols.div_ceil(threads_per_block);

        let stream = Self::_get_stream();

        unsafe {
            // 4. Launch the kernel
            // Note: We pass num_rows and num_cols as i32 to match your C++ signature
            let _ = launch!(sum_kernel<<< grid_size, threads_per_block, 0, stream >>>(
                self.device_buffer.as_device_ptr(),
                result.as_device_ptr(),
                num_rows as i32,
                num_cols as i32
            ));
        }

        // 5. Return a new Self with the reduced shape (e.g., [1, cols] or just [cols])
        Ok(Self::_with_device_buffer(vec![1, num_cols], result))
    }

    fn check_shape(shape: &[u32]) -> Option<Result<GpuTensor<T>, String>> {
        if shape.is_empty() {
            return Some(Err(
                "ShapeError: GpuTensor must have at least one dimension.".to_string(),
            ));
        }

        if shape.len() > 2 {
            return Some(Err(
                "TemporaryShapeRestriction: Currently only accepting tensors upto 2 dimensions"
                    .to_string(),
            ));
        }
        None
    }

    fn calculate_length(shape: &Vec<u32>) -> u32 {
        let mut size = 1;

        for i in shape {
            size *= i;
        }
        size
    }

    fn _get_initialized_buffer(&self, size: usize) -> CustomDeviceBuffer<T> {
        let buffer = get_device_buffer(size);

        let threads_per_block = 1024;
        let grid_1d = (size as u32).div_ceil(threads_per_block);
        let stream = Self::_get_stream();
        let operation = Self::_get_function("fill_value");

        unsafe {
            let _ = launch!(operation<<<grid_1d, threads_per_block, 0, stream>>>(
                buffer.as_device_ptr(),
                size as i32,
                T::zero()
            ));
        }

        buffer
    }

    fn _new(shape: Vec<u32>, data: Vec<T>) -> Result<Self, String> {
        if let Some(value) = Self::check_shape(&shape) {
            return value;
        }

        let size = Self::calculate_length(&shape);

        if size != (data.len() as u32) {
            let err = format!("DataError: Data length ({}) does not match total num of elements provided by dimensions ({}))", data.len(), size);
            return Err(err);
        }

        Ok(Self::_with_device_buffer(
            shape.to_vec(),
            get_device_buffer_from_slice(&data),
        ))
    }

    fn _new_with_value(shape: Vec<u32>, value: T) -> Result<Self, String> {
        if let Some(value) = Self::check_shape(&shape) {
            return value;
        }

        let size = Self::calculate_length(&shape) as usize;

        let device_buffer = get_device_buffer::<T>(size);

        let threads_per_block = 1024; // Use a typical 1D block size
        let grid_1d = (size as u32).div_ceil(threads_per_block);

        let stream = Self::_get_stream();

        let operation = Self::_get_function("fill_value");

        unsafe {
            let _ = launch!(operation<<<grid_1d, threads_per_block, 0, stream>>>(
                device_buffer.as_device_ptr(),
                size as i32,
                value
            ));
        };

        Ok(Self::_with_device_buffer(shape.to_vec(), device_buffer))
    }

    fn _with_device_buffer(shape: Vec<u32>, device_buffer: CustomDeviceBuffer<T>) -> Self {
        Self {
            shape: shape.clone(),
            device_buffer,
        }
    }

    fn _gpu_mul(&self, rhs: &Self) -> Result<Self, String> {
        // Set up common block size
        let block_dim = 16;

        // Calculate grid size using ceiling division
        let grid_x = rhs.shape[1].div_ceil(block_dim);
        let grid_y = self.shape[0].div_ceil(block_dim);

        let total_elements = (self.shape[0] * rhs.shape[1]) as usize;

        let result = get_device_buffer(total_elements);

        let stream = Self::_get_stream();

        let mat_mul = Self::_get_function("matrix_mul");

        unsafe {
            let _ = launch!(mat_mul<<<(grid_x, grid_y, 1), (block_dim, block_dim, 1), 0, stream>>>(
                self.device_buffer.as_device_ptr(),
                rhs.device_buffer.as_device_ptr(),
                result.as_device_ptr(),
                self.shape[0] as i32,
                rhs.shape[1] as i32,
                self.shape[1] as i32
            ));
        }

        let result_shape = vec![self.shape[0], rhs.shape[1]];
        Ok(Self::_with_device_buffer(result_shape, result))
    }

    fn _gpu_mul_cublas(&self, rhs: &Self) -> Result<Self, String> {
        let m = self.shape[0] as i32;
        let k = self.shape[1] as i32;
        let n = rhs.shape[1] as i32;

        let total_elements = (m * n) as usize;
        let result = get_device_buffer(total_elements);

        let alpha = T::one();
        let beta = T::zero();

        unsafe {
            cublasSgemm_v2(
                Self::_get_cublas_handle(),
                cublasOperation_t::CUBLAS_OP_N,
                cublasOperation_t::CUBLAS_OP_N,
                n,
                m,
                k,
                &alpha.f32(),
                rhs.device_buffer.as_device_ptr().as_raw() as *const f32,
                n,
                self.device_buffer.as_device_ptr().as_raw() as *const f32,
                k,
                &beta.f32(),
                result.as_device_ptr().as_raw() as *mut f32,
                n,
            );
        }

        let result_shape = vec![self.shape[0], rhs.shape[1]];
        Ok(Self::_with_device_buffer(result_shape, result))
    }

    fn _mul(&self, rhs: &Self) -> Result<Self, String> {
        if self.shape[1] != rhs.shape[0] {
            let s = format!(
                "ShapeMismatch:The dimensions of two matrices are not compatible for multiplication- {:?} {:?}",
                self.shape, rhs.shape
            );
            return Err(s);
        }

        self._gpu_mul_cublas(rhs)
    }
}

impl<T> TensorMath<T> for GpuTensor<T>
where
    T: FloatingPoint + Zeroable + DeviceCopy,
{
    type MathOutputElem = T;
    type MathOutput = GpuTensor<T>;

    fn sigmoid(&self) -> Result<Self::MathOutput, String> {
        self.element_op(OpType::Sigmoid)
    }

    fn log(&self) -> Result<Self::MathOutput, String> {
        self.element_op(OpType::Log)
    }

    fn ln(&self) -> Result<Self::MathOutput, String> {
        self.element_op(OpType::Ln)
    }

    fn sin(&self) -> Result<Self::MathOutput, String> {
        self.element_op(OpType::Sin)
    }

    fn cos(&self) -> Result<Self::MathOutput, String> {
        self.element_op(OpType::Cos)
    }

    fn tan(&self) -> Result<Self::MathOutput, String> {
        self.element_op(OpType::Tan)
    }

    fn tanh(&self) -> Result<Self::MathOutput, String> {
        self.element_op(OpType::Tanh)
    }

    fn exp(&self) -> Result<Self::MathOutput, String> {
        self.element_op(OpType::Exp)
    }

    fn relu(&self) -> Result<Self::MathOutput, String> {
        self.element_op(OpType::ReLU)
    }

    fn greater_than_zero_mask(&self) -> Result<Self::MathOutput, String> {
        self.element_op(OpType::GreaterThanZeroMask)
    }
}
