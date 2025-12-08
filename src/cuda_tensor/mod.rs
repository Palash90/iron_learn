use cust::launch;
use cust::memory::bytemuck::Zeroable;
use cust::module::Module;
use cust::stream::Stream;
use cust::stream::StreamFlags;
use cust::{device, memory::CopyDestination, prelude::DeviceBuffer};

use crate::init_context;
use crate::numeric::{Numeric, SignedNumeric};
use std::ops::{Add, Mul, Neg, Sub};

#[derive(Debug)]
pub struct GpuTensor<T: Numeric> {
    module: Module,
    shape: Vec<u32>,
    device_buffer: DeviceBuffer<T>,
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
                print!("{}\t", self.get_data()[r * cols + c]);
            }
            println!();
        }
    }
}

impl<T: Numeric + Zeroable> GpuTensor<T> {
    pub fn _exp(operand: &Self) {
        unimplemented!("Tensor Exponential unimplemented");
    }

    fn _t(&self) -> Result<Self, String> {
        if self.shape.len() > 2 {
            return Err("Only 2D tensors can be transposed.".to_string());
        }

        let new_shape = vec![self.shape[1], self.shape[0]];

        unimplemented!("Tensor Transpose Unimplemented");
    }

    fn _data(&self) -> Vec<T> {
        let total_elements = match self.shape.len() {
            2 => (self.shape[0] * self.shape[1]) as usize,
            _ => self.shape[0] as usize,
        };

        let mut data = vec![T::zero(); (total_elements) as usize];

        self.device_buffer.copy_to(&mut data);
        data
    }

    fn _shape(&self) -> Vec<u32> {
        self.shape.clone()
    }

    fn _add(&self, rhs: &Self, sub: bool) -> Result<Self, String> {
        if self.shape != rhs.shape {
            return Err(format!("ShapeMismatch:The dimensions of two matrices are not compatible for addition/subtraction- {:?} {:?}", self.shape, rhs.shape));
        }

        let add = match self.module.get_function("vector_add") {
            Ok(m) => m,
            Err(e) => return Err(e.to_string()),
        };

        let mut total_elements;
        if self.shape.len() == 2 {
            total_elements = (self.shape[0] * self.shape[1]) as usize;
        } else {
            total_elements = self.shape[0] as usize
        }

        let result = DeviceBuffer::<T>::zeroed(total_elements).unwrap();

        let total_size_u32 = total_elements as u32;
        let threads_per_block = 1024;

        let grid_1d = (total_size_u32 + threads_per_block - 1) / threads_per_block;

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();
        unsafe {
            launch!(add<<< (grid_1d, 1, 1), 1024, 0, stream >>>(
                self.device_buffer.as_device_ptr(),
                rhs.device_buffer.as_device_ptr(),
                result.as_device_ptr(),
                total_size_u32 as i32,
                0
            ));
        }

        stream.synchronize();

        Ok(Self::_with_device_buffer(self.shape.clone(), result))
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

        let ptx = include_str!("../../kernels/gpu_kernels.ptx");
        let module = Module::from_ptx(ptx, &[]).expect("CUDA module could not be initiated");

        match DeviceBuffer::from_slice(&data) {
            Ok(device_buffer) => Ok(Self {
                module: module,
                shape: shape.clone(),
                device_buffer,
            }),
            Err(_) => Err("CUDA Error".to_string()),
        }
    }

    fn _with_device_buffer(shape: Vec<u32>, device_buffer: DeviceBuffer<T>) -> Self {
        let ptx = include_str!("../../kernels/gpu_kernels.ptx");
        let module = Module::from_ptx(ptx, &[]).expect("CUDA module could not be initiated");

        Self {
            module: module,
            shape: shape.clone(),
            device_buffer,
        }
    }

    pub fn hadamard(&self, rhs: &Self) -> Result<Self, String> {
        unimplemented!();
    }

    fn _gpu_mul(&self, rhs: &Self) -> Result<Self, String> {
        // Set up common block size
        let block_dim = 16;

        // Calculate grid size using ceiling division
        let grid_x = (rhs.shape[1] + block_dim - 1) / block_dim;
        let grid_y = (self.shape[0] + block_dim - 1) / block_dim;

        let total_elements = self.shape[0] * rhs.shape[1];

        let result = DeviceBuffer::<T>::zeroed(total_elements as usize).unwrap();

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

        let mat_mul = match self.module.get_function("matrixMul") {
            Ok(m) => m,
            Err(e) => return Err(e.to_string()),
        };

        unsafe {
            launch!(mat_mul<<<(grid_x, grid_y, 1), (block_dim, block_dim, 1), 0, stream>>>(
                self.device_buffer.as_device_ptr(),
                rhs.device_buffer.as_device_ptr(),
                result.as_device_ptr(),
                self.shape[0] as i32,
                rhs.shape[1] as i32,
                self.shape[1] as i32
            ));
        }

        stream.synchronize();
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

        self._gpu_mul(rhs)
    }

    fn _s(&self, scalar: T) -> Result<Self, String> {
        unimplemented!();
    }
}

impl<T: Numeric + Zeroable> GpuTensor<T> {
    pub fn exp(operand: &Self) {
        Self::_exp(operand)
    }

    pub fn new(shape: Vec<u32>, data: Vec<T>) -> Result<Self, String> {
        Self::_new(shape, data)
    }

    pub fn get_data(&self) -> Vec<T> {
        self._data()
    }

    pub fn get_shape(&self) -> Vec<u32> {
        self._shape()
    }

    pub fn add(&self, rhs: &Self) -> Result<Self, String> {
        self._add(rhs, false)
    }
    pub fn sum(&self) -> Result<Self, String> {
        let mut sum_vector = vec![T::zero(); self.shape[1] as usize];
        let rows = self.shape[0] as usize;
        let cols = self.shape[1] as usize;

        unimplemented!();
    }

    pub fn sub(&self, rhs: &Self) -> Result<Self, String> {
        self._add(rhs, true)
    }

    pub fn mul(&self, rhs: &Self) -> Result<Self, String> {
        self._mul(rhs)
    }

    pub fn multiply(&self, rhs: &Self) -> Result<Self, String> {
        self.hadamard(rhs)
    }

    pub fn t(&self) -> Result<Self, String> {
        self._t()
    }

    pub fn scale(&self, scalar: T) -> Result<Self, String> {
        self._s(scalar)
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
        unimplemented!();
    }
}

impl<T: Numeric + Zeroable> PartialEq for GpuTensor<T> {
    fn eq(&self, other: &Self) -> bool {
        // 1. Shape Check
        if self.shape != other.shape {
            return false;
        }

        // 2. Calculate total elements robustly (FIXED)
        // Use the total product of the shape elements, which is correct for 1D or 2D.
        let element_count = self.shape.iter().product::<u32>();

        if element_count == 0 {
            return true; // Two empty tensors are equal
        }

        let total_size = element_count as u64;
        let threads_per_block = 1024;

        let grid_1d = (element_count + threads_per_block - 1) / threads_per_block;

        // 3. Get the kernel function
        let compare = match self.module.get_function("compareMemory") {
            Ok(m) => m,
            Err(e) => {
                eprintln!("Error getting kernel 'compareMemory': {}", e);
                return false;
            }
        };

        // 4. Prepare host and device result buffers
        // result_host starts at 1 (equal). Kernel sets to 0 (not equal) on mismatch.
        let mut result_host = [1i32];
        let mut result_device = match DeviceBuffer::from_slice(&result_host) {
            Ok(device_buf) => device_buf,
            Err(e) => {
                eprintln!("Error creating device buffer for result: {:?}", e);
                return false;
            }
        };

        // 5. Create stream
        let stream = match Stream::new(StreamFlags::NON_BLOCKING, None) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error creating stream: {}", e);
                return false;
            }
        };

        // 6. Launch the kernel
        unsafe {
            launch!(compare<<<(grid_1d, 1, 1), threads_per_block, 0, stream>>>(
                self.device_buffer.as_device_ptr(),
                other.device_buffer.as_device_ptr(),
                total_size, // Use the correct calculated size
                result_device.as_device_ptr(),
            ));
        }

        // 7. Synchronize stream
        if let Err(e) = stream.synchronize() {
            eprintln!("Error synchronizing stream: {}", e);
            return false;
        }

        // 8. Copy result back from GPU to CPU (CRITICAL FIX)
        match result_device.copy_to(&mut result_host) {
            Ok(_) => {
                println!("Copied from device to host {:?}", result_host);
                // If the value copied back is 1, they are equal.
                result_host[0] == 1
            }
            Err(e) => {
                eprintln!("Error copying result from device to host: {}", e);
                false
            }
        }
    }
}

#[cfg(test)]
#[test]
fn test_new() {
    match cust::quick_init() {
        Ok(context) => {
            eprintln!("✓ GPU initialization successful");
            init_context("Iron Learn", 5, String::new(), 0.0, 0, true, Some(context));
        }
        Err(e) => {
            return;
        }
    };

    let t = GpuTensor::new(vec![1u32, 2u32], vec![1i8, 2i8]).unwrap();

    println!("GPU tensor got created");

    assert_eq!(t.shape, vec![1u32, 2u32]);
    assert_eq!(t.get_data(), vec![1i8, 2i8]);
}
