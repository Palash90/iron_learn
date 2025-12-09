use crate::init_context;
use crate::numeric::{Numeric, SignedNumeric};
use cust::memory::bytemuck::Zeroable;
use cust::module::Module;
use cust::prelude::DeviceBuffer;
use std::ops::{Add, Mul, Neg, Sub};

mod tensor_ops;
use tensor_ops::OpType;

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
    pub fn sin(&self) -> Result<Self, String> {
        self.element_op(OpType::SIN, T::zero())
    }
    pub fn exp(&self) -> Result<Self, String> {
        self.element_op(OpType::EXP, T::zero())
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

        let data = self.get_data();

        for r in 0..rows {
            for c in 0..cols {
                sum_vector[c] = sum_vector[c] + data[r * cols + c];
            }
        }

        Self::new(vec![1, self.shape[1]], sum_vector)
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
        self.element_op(OpType::SCALE, scalar)
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
        self._eq(other)
    }
}

#[cfg(test)]
#[test]
fn test_new() {
    match cust::quick_init() {
        Ok(context) => {
            println!("✓ GPU initialization successful");
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
