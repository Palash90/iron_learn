use crate::cuda_tensor::custom_device_buffer::{
    get_device_buffer, get_device_buffer_from_slice, CustomDeviceBuffer,
};
use crate::numeric::{Numeric, SignedNumeric};
use crate::tensor::math::TensorMath;
use crate::GPU_CONTEXT;
use core::ffi::c_void;
use cublas_sys::*;
use cust::memory::bytemuck::Zeroable;
use cust::memory::CopyDestination;
use cust::sys::*;
use std::ops::{Add, Mul, Neg, Sub};

//mod tensor_ops;
use crate::Tensor;
mod custom_device_buffer;
mod mem_pool;
pub use mem_pool::CudaMemoryPool;
mod cublas_handle;
pub use cublas_handle::CublasHandle;
use cust::launch;
use cust::prelude::DeviceBuffer;
use cust::prelude::Function;
use cust::stream::Stream;

#[derive(Clone, Copy, Debug)]
enum OpType {
    EXP = 0,
    SCALE = 1,
    SIN = 2,
    COS = 3,
    TAN = 4,
    TANH = 5,
    SIGMOID = 6,
    LOG = 7,
    LN = 8,
}

#[derive(Debug)]
pub struct GpuTensor<T: Numeric> {
    shape: Vec<u32>,
    device_buffer: CustomDeviceBuffer<T>,
}

impl<T: Numeric + Zeroable> GpuTensor<T>
where
    T: Numeric,
{
    pub fn print_matrix(&self) {
        let rows = self.shape[0] as usize;
        let cols = match self.shape.len() {
            2 => self.shape[1] as usize,
            _ => 1,
        };

        for r in 0..rows {
            for c in 0..cols {
                print!("{:.2?}\t", self.get_data()[r * cols + c]);
            }
            println!();
        }
    }
}

impl<T: Numeric + Zeroable> Tensor<T> for GpuTensor<T> {
    fn new(shape: Vec<u32>, data: Vec<T>) -> Result<Self, String> {
        Self::_new(shape, data)
    }

    fn get_data(&self) -> Vec<T> {
        self._data()
    }

    fn get_shape(&self) -> &Vec<u32> {
        &self.shape
    }

    fn add(&self, rhs: &Self) -> Result<Self, String> {
        self._add(rhs, false)
    }
    fn sum(&self) -> Result<Self, String> {
        self._sum()
    }

    fn sub(&self, rhs: &Self) -> Result<Self, String> {
        self._add(rhs, true)
    }

    fn mul(&self, rhs: &Self) -> Result<Self, String> {
        self._mul(rhs)
    }

    fn multiply(&self, rhs: &Self) -> Result<Self, String> {
        self.hadamard(rhs)
    }

    fn t(&self) -> Result<Self, String> {
        self._t()
    }

    fn scale(&self, scalar: T) -> Result<Self, String> {
        self.element_op(OpType::SCALE, scalar)
    }

    fn synchronize() {
        let _ = &(GPU_CONTEXT
            .get()
            .expect("No GPU Context Intialized")
            .stream
            .as_ref()
            .expect("Stream could not be found"))
        .synchronize();
    }

    fn print_matrix(&self) -> () {
        self.print_matrix()
    }

    fn zeroes(shape: &Vec<u32>) -> Self {
        Self::_new_with_value(shape.to_vec(), T::zero()).unwrap()
    }

    fn ones(shape: &Vec<u32>) -> Self {
        Self::_new_with_value(shape.to_vec(), T::one()).unwrap()
    }
}

impl<T: Numeric + Zeroable> Add for GpuTensor<T> {
    type Output = Result<Self, String>;

    fn add(self, rhs: Self) -> Result<Self, String> {
        self._add(&rhs, false)
    }
}

impl<T: Numeric + Zeroable> Sub for GpuTensor<T> {
    type Output = Result<Self, String>;

    fn sub(self, rhs: Self) -> Result<Self, String> {
        self._add(&rhs, true)
    }
}

impl<T: Numeric + Zeroable> Mul for GpuTensor<T> {
    type Output = Result<Self, String>;

    fn mul(self, rhs: Self) -> Result<Self, String> {
        self._mul(&rhs)
    }
}

impl<T: SignedNumeric + Zeroable> Neg for GpuTensor<T> {
    type Output = Result<Self, String>;
    fn neg(self) -> Result<Self, String> {
        self.element_op(OpType::SCALE, -T::one())
    }
}

impl<T: Numeric + Zeroable> PartialEq for GpuTensor<T> {
    fn eq(&self, other: &Self) -> bool {
        println!("Partial eq called");
        self._eq(other)
    }
}

impl<T: Numeric + Zeroable> GpuTensor<T> {
    fn _get_function(fn_name: &str) -> Function {
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
            .expect(&format!("Stream not found"))
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
            if self.get_data()[i] != other.get_data()[i] {
                return false;
            }
        }
        return true;
    }

    fn element_op(&self, op_type: OpType, scale: T) -> Result<Self, String> {
        let block_dim = 16;

        let total_elements: u32 = self.shape.iter().product();
        let threads_per_block = 1024; // Use a typical 1D block size
        let grid_1d = (total_elements + threads_per_block - 1) / threads_per_block;

        let result = get_device_buffer(total_elements as usize);

        let stream = Self::_get_stream();

        let operation = Self::_get_function("element_op");

        unsafe {
            let _ = launch!(operation<<<grid_1d, threads_per_block, 0, stream>>>(
                self.device_buffer.as_device_ptr(),
                result.as_device_ptr(),
                total_elements as i32,
                op_type as i32,
                scale
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

        let result = get_device_buffer(self.shape.iter().product::<u32>() as usize);
        let new_shape = vec![self.shape[1], self.shape[0]];

        const BLOCK_DIM_X: u32 = 16;
        const BLOCK_DIM_Y: u32 = 16;

        let m = self.shape[0] as i32;
        let n = self.shape[1] as i32;

        let grid_x = (n as u32 + BLOCK_DIM_X - 1) / BLOCK_DIM_X;
        let grid_y = (m as u32 + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y;

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

        let mut data = vec![T::zero(); (total_elements) as usize];

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

    fn _add(&self, rhs: &Self, sub: bool) -> Result<Self, String> {
        if self.shape != rhs.shape {
            return Err(format!("ShapeMismatch:The dimensions of two matrices are not compatible for addition/subtraction- {:?} {:?}", self.shape, rhs.shape));
        }

        let add = Self::_get_function("vector_add");

        let total_elements = self.shape.iter().product::<u32>() as usize;

        let result = get_device_buffer(total_elements);

        let total_size_u32 = total_elements as u32;
        let threads_per_block = 1024;

        let grid_1d = (total_size_u32 + threads_per_block - 1) / threads_per_block;
        let sub_int = if sub { 1i32 } else { 0i32 };

        let stream = Self::_get_stream();
        unsafe {
            let _ = launch!(add<<< grid_1d, threads_per_block, 0, stream >>>(
                self.device_buffer.as_device_ptr(),
                rhs.device_buffer.as_device_ptr(),
                result.as_device_ptr(),
                total_size_u32 as i32,
                sub_int
            ));
        }

        Ok(Self::_with_device_buffer(self.shape.clone(), result))
    }

    fn _sum(&self) -> Result<Self, String> {
        let sum = Self::_get_function("column_reduce");

        let total_elements = self.shape.iter().product::<u32>() as usize;

        let result = get_device_buffer(total_elements);

        let total_size_u32 = total_elements as u32;
        let threads_per_block = 1024;

        let grid_1d = (total_size_u32 + threads_per_block - 1) / threads_per_block;

        let stream = Self::_get_stream();
        unsafe {
            let _ = launch!(sum<<< (grid_1d, 1, 1), 1024, 0, stream >>>(
                self.device_buffer.as_device_ptr(),
                result.as_device_ptr(),
                self.shape[0],
                self.shape[1]
            ));
        }

        Ok(Self::_with_device_buffer(vec![1, self.shape[1]], result))
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

        let block_dim = 16;

        let threads_per_block = 1024; // Use a typical 1D block size
        let grid_1d = (size as u32 + threads_per_block - 1) / threads_per_block;

        let stream = Self::_get_stream();

        let operation = Self::_get_function("fill_value");

        unsafe {
            let _ = launch!(operation<<<(grid_1d, 1, 1), (block_dim, block_dim, 1), 0, stream>>>(
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

    fn hadamard(&self, rhs: &Self) -> Result<Self, String> {
        if self.shape != rhs.shape {
            let s = format!(
                "ShapeMismatch:The dimensions of two matrices are not compatible for hadamard product- {:?} {:?}",
                self.shape, rhs.shape
            );
            return Err(s);
        }

        // Set up common block size
        let block_dim = 16;

        let total_elements = self.shape.iter().product::<u32>();

        let threads_per_block = 1024; // Use a typical 1D block size
        let grid_1d = (total_elements + threads_per_block - 1) / threads_per_block as u32;

        let result = get_device_buffer(total_elements as usize);

        let stream = Self::_get_stream();

        let hadamard = Self::_get_function("hadamard_prod");

        unsafe {
            let _ = launch!(hadamard<<<(grid_1d, 1, 1), (block_dim, block_dim, 1), 0, stream>>>(
                self.device_buffer.as_device_ptr(),
                rhs.device_buffer.as_device_ptr(),
                result.as_device_ptr(),
                total_elements as i32
            ));
        }

        Ok(Self::_with_device_buffer(self.shape.clone(), result))
    }

    fn _gpu_mul(&self, rhs: &Self) -> Result<Self, String> {
        // Set up common block size
        let block_dim = 16;

        // Calculate grid size using ceiling division
        let grid_x = (rhs.shape[1] + block_dim - 1) / block_dim;
        let grid_y = (self.shape[0] + block_dim - 1) / block_dim;

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
    T: Numeric + Zeroable,
{
    type MathOutput = GpuTensor<T>;

    fn sigmoid(&self) -> Result<Self::MathOutput, String> {
        self.element_op(OpType::SIGMOID, T::zero())
    }

    fn log(&self) -> Result<Self::MathOutput, String> {
        self.element_op(OpType::LOG, T::zero())
    }

    fn ln(&self) -> Result<Self::MathOutput, String> {
        self.element_op(OpType::LN, T::zero())
    }

    fn sin(&self) -> Result<Self::MathOutput, String> {
        self.element_op(OpType::SIN, T::zero())
    }

    fn cos(&self) -> Result<Self::MathOutput, String> {
        self.element_op(OpType::COS, T::zero())
    }

    fn tan(&self) -> Result<Self::MathOutput, String> {
        self.element_op(OpType::TAN, T::zero())
    }

    fn tanh(&self) -> Result<Self::MathOutput, String> {
        self.element_op(OpType::TANH, T::zero())
    }

    fn exp(&self) -> Result<Self::MathOutput, String> {
        self.element_op(OpType::EXP, T::zero())
    }
}
