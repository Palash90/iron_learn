//! The tensor module defines necessary methods for implementing the linear algebra necessary for machine learning

use std::ops::{Add, Mul};

use crate::numeric::Numeric;

/// Tensor is the heart of this whole library. This provides the basic mathematical operations.

#[derive(Debug, PartialEq)]
pub struct Tensor<T: Numeric> {
    shape: Vec<u32>,
    data: Vec<T>,
}

impl<T: Numeric> Tensor<T> {
    pub fn new(shape: Vec<u32>, data: Vec<T>) -> Result<Self, String> {
        if shape.len() == 0 || shape.len() > 2 {
            return Err(format!(
                "TemporaryShapeRestriction: Currently only accepting tensors upto 1 - 2 dimensions"
            ));
        }

        let mut size = 1;

        for i in &shape {
            size *= i;
        }

        if size != data.len().try_into().unwrap() {
            let err = String::from(format!("DataError: Data length ({}) does not match total num of elements provided by dimensions ({}))", data.len(), size));
            return Err(err);
        }

        Ok(Self { shape, data })
    }

    pub fn get_data(self) -> Vec<T> {
        self.data
    }

    pub fn multiply(self, rhs: Self) -> Result<Self, &'static str> {
        let mut result = Vec::with_capacity(self.data.len());

        if self.shape != rhs.shape {
            return Err("ShapeMismatch: Mismatch in shape of two Tensors.");
        }

        for i in 0..self.data.len() {
            result.push(self.data[i] * rhs.data[i])
        }

        Ok(Self {
            shape: self.shape,
            data: result,
        })
    }
}

impl<T: Numeric> Add for Tensor<T> {
    type Output = Result<Self, &'static str>;

    fn add(self, rhs: Self) -> Result<Self, &'static str> {
        let mut result = Vec::with_capacity(self.data.len());

        if self.shape != rhs.shape {
            return Err("ShapeMismatch: Mismatch in shape of two Tensors.");
        }

        for i in 0..self.data.len() {
            result.push(self.data[i] + rhs.data[i])
        }

        Ok(Self {
            shape: self.shape,
            data: result,
        })
    }
}

impl<T: Numeric> Mul for Tensor<T> {
    type Output = Result<Self, String>;

    fn mul(self, rhs: Self) -> Result<Self, String> {
        if self.shape[1] != rhs.shape[0] {
            let s = format!(
                "ShapeMismatch:The dimensions of two matrices are not matching {:?} {:?}",
                self.shape, rhs.shape
            );
            return Err(s);
        }

        let rows = self.shape[0] as usize;
        let cols = rhs.shape[1] as usize;
        let common_dim = self.shape[1] as usize;

        let mut data = vec![T::zero(); rows * cols];

        for i in 0..rows {
            for j in 0..cols {
                for k in 0..common_dim {
                    let val = self.data[i * common_dim + k] * rhs.data[k * cols + j];
                    data[i * cols + j] = data[i * cols + j] + val;
                }
            }
        }

        Ok(Tensor {
            shape: vec![rows as u32, cols as u32],
            data,
        })
    }
}

#[cfg(test)]
#[test]
fn test_new() {
    let t = Tensor::new(vec![1u32, 2u32], vec![1i8, 2i8]).unwrap();

    assert_eq!(t.shape, vec![1u32, 2u32]);
    assert_eq!(t.data, vec![1i8, 2i8]);
}
