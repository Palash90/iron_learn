mod display;
use crate::{GLOBAL_CONTEXT, numeric::{Numeric, SignedNumeric}};
use std::ops::{Add, Mul, Neg, Sub};

mod cpu_tensor;
mod tensor_backend;

use cpu_tensor::CpuTensor;
use tensor_backend::TensorBackend;

/// Enum to represent different Tensor backends (CPU and GPU)
pub enum TensorFamily<T: Numeric> {
    Cpu(CpuTensor<T>),
}

// Implement the core operations for TensorFamily by delegating to the backend
impl<T: Numeric> TensorFamily<T> {
    // This helper method delegates the call to the appropriate inner backend
    fn dispatch<F>(&self, rhs: &Self, op: F) -> Result<Self, String>
    where
        F: Fn(&dyn TensorBackend<T>, &dyn TensorBackend<T>) -> Result<Box<dyn TensorBackend<T>>, String>,
    {
        // This match ensures we are operating on the same family type.
        // It also handles creating the output TensorFamily type.
        match (self, rhs) {
            (TensorFamily::Cpu(l), TensorFamily::Cpu(r)) => {
                let result = op(l, r)?; // l and r are &CpuTensor<T>
                Ok(TensorFamily::Cpu(
                    *result.downcast::<CpuTensor<T>>().expect("Backend mismatch"),
                ))
            }
            
            _ => Err("Operation attempted on Tensors with different backends (e.g., CPU + GPU)".to_string()),
        }
    }
}




#[derive(Debug, PartialEq)]
pub struct Tensor<T: Numeric> {
    backend: TensorFamily<T>,
}
impl<T> Tensor<T>
where
    T: Numeric,
{
    pub fn print_matrix(&self) {
        let rows = self.shape[0] as usize;
        let cols = self.shape[1] as usize;

        for r in 0..rows {
            for c in 0..cols {
                print!("{}\t", self.data[r * cols + c]);
            }
            println!();
        }
    }
}



// The public API of Tensor type
impl<T: Numeric> Tensor<T> {
    
    pub fn exp(operand: &Self) -> Tensor<f64> {
        // Delegates to the backend implementation
        None
    }

    
    pub fn new(shape: Vec<u32>, data: Vec<T>) -> Result<Self, String> {
        let ctx = GLOBAL_CONTEXT.get().expect("Context not initialized");

        let backend = match ctx.gpu_enabled {
            false => TensorFamily::Cpu(CpuTensor::from_data(shape, data)?),
            true => unimplemented!("GPU backend is not yet implemented"),
        };

        Ok(Self { backend })
    }

    pub fn get_data(&self) -> Vec<T> {
        // Delegates to the backend implementation
        vec![]
    }

    
    pub fn get_shape(&self) -> Vec<u32> {
        // Delegates to the backend implementation
    }

    pub fn add(&self, rhs: &Self) -> Result<Tensor<T>, String> {
        // Delegates to the backend implementation
    }

    pub fn sub(&self, rhs: &Self) -> Result<Tensor<T>, String> {
        // Delegates to the backend implementation
    }

    pub fn mul(&self, rhs: &Self) -> Result<Tensor<T>, String> {
        // Delegates to the backend implementation
    }

    pub fn multiply(&self, rhs: &Self) -> Result<Self, String> {
        // Delegates to the backend implementation
    }

    pub fn t(&self) -> Result<Self, String> {
       // Delegates to the backend implementation
    }

    pub fn scale(&self, scalar: T) -> Self {
        // Delegates to the backend implementation
    }
}

impl<T: Numeric> Add for Tensor<T> {
    type Output = Result<Self, String>;

    fn add(self, rhs: Self) -> Result<Self, String> {
        // Delegates to the backend implementation
    }
}

impl<T: Numeric> Sub for Tensor<T> {
    type Output = Result<Self, String>;

    fn sub(self, rhs: Self) -> Result<Self, String> {
        // Delegates to the backend implementation
    }
}

impl<T: Numeric> Mul for Tensor<T> {
    type Output = Result<Self, String>;

    fn mul(self, rhs: Self) -> Result<Self, String> {
        // Delegates to the backend implementation
    }
}

impl<T: SignedNumeric> Neg for Tensor<T> {
    type Output = Self;
    fn neg(self) -> Self {
        let result: Vec<T> = self.data.iter().map(|t| -*t).collect();
        Tensor::new(self.shape, result).unwrap()
    }
}

