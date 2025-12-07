use cust::{device, memory::CopyDestination, prelude::DeviceBuffer};

use crate::init_context;
use crate::numeric::{Numeric, SignedNumeric};
use std::ops::{Add, Mul, Neg, Sub};

#[derive(Debug)]
pub struct GpuTensor<T: Numeric> {
    shape: Vec<u32>,
    device_buffer: DeviceBuffer<T>,
}
impl<T> GpuTensor<T>
where
    T: Numeric,
{
    pub fn print_matrix(&self) {
        let rows = self.shape[0] as usize;
        let cols = self.shape[1] as usize;

        for r in 0..rows {
            for c in 0..cols {
                print!("{}\t", self.get_data()[r * cols + c]);
            }
            println!();
        }
    }
}

impl<T: Numeric> GpuTensor<T> {
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
        let mut data = vec![T::zero(); (self.shape[0] * self.shape[1]) as usize];

        self.device_buffer.copy_to(&mut data);
        data
    }

    fn _shape(&self) -> Vec<u32> {
        self.shape.clone()
    }

    fn _add(&self, rhs: &Self, sub: bool) -> Result<Self, String> {
        unimplemented!("Add Sub");
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

        match DeviceBuffer::from_slice(&data) {
            Ok(device_buffer) => Ok(Self {
                shape,
                device_buffer,
            }),
            Err(_) => Err("CUDA Error".to_string()),
        }
    }

    pub fn hadamard(&self, rhs: &Self) -> Result<Self, String> {
        unimplemented!();
    }

    fn _gpu_mul(&self, rhs: &Self) -> Result<Self, String> {
        unimplemented!("Need to call GemmV");
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

impl<T: Numeric> GpuTensor<T> {
    pub fn to_device() {}

    pub fn from_device() {}

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

impl<T: Numeric> Add for GpuTensor<T> {
    type Output = Result<Self, String>;

    fn add(self, rhs: Self) -> Result<Self, String> {
        self._add(&rhs, false)
    }
}

impl<T: Numeric> Sub for GpuTensor<T> {
    type Output = Result<Self, String>;

    fn sub(self, rhs: Self) -> Result<Self, String> {
        self._add(&rhs, true)
    }
}

impl<T: Numeric> Mul for GpuTensor<T> {
    type Output = Result<Self, String>;

    fn mul(self, rhs: Self) -> Result<Self, String> {
        self._mul(&rhs)
    }
}

impl<T: SignedNumeric> Neg for GpuTensor<T> {
    type Output = Result<Self, String>;
    fn neg(self) -> Result<Self, String> {
        unimplemented!();
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
