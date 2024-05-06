use crate::numeric::Numeric;
use crate::tensor::Tensor;
use std::ops;

impl<T: Numeric> ops::Add for Tensor<T> {
    type Output = Result<Self, &'static str>;

    fn add(self, rhs: Self) -> Result<Self, &'static str> {
        let mut result_vector = Vec::with_capacity(self.data.len());

        if self.dimensions != rhs.dimensions {
            return Err("Incompatible Dimensions: Mismatch in dimension of two Tensors.");
        }

        if self.data.len() != rhs.data.len() {
            return Err("Element mismatch in two tensors.");
        }

        for i in 0..self.data.len() {
            result_vector.push(self.data[i] + rhs.data[i])
        }

        Ok(Self {
            dimensions: self.dimensions,
            data: result_vector,
        })
    }
}
