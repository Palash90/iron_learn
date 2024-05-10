use crate::numeric::Numeric;
use crate::tensor::Tensor;
use std::ops::Add;
use std::ops::Mul;

#[derive(Debug, PartialEq)]
pub struct Vector<T: Numeric> {
    tensor: Tensor<T>,
}

impl<T: Numeric> Vector<T> {
    pub fn new(shape: Vec<u32>, data: Vec<T>) -> Result<Self, String> {
        if shape.len() != 1 {
            return Err(format!(
                "VectorShapeError: Vector can have only one dimension, provided {}",
                shape.len()
            ));
        }

        let t = Tensor::new(shape, data);

        Ok(Self { tensor: t? })
    }
}

impl<T: Numeric> Add for Vector<T> {
    type Output = Result<Self, &'static str>;

    fn add(self, rhs: Self) -> Result<Self, &'static str> {
        let result = self.tensor + rhs.tensor;
        Ok(Self { tensor: result? })
    }
}

impl<T: Numeric> Mul for Vector<T> {
    type Output = Result<T, String>;

    fn mul(self, rhs: Self) -> Result<T, String> {
        let h = self.tensor.multiply(rhs.tensor)?;

        let mut result: T = T::zero();

        for i in h.get_data() {
            result = result + i;
        }

        Ok(result)
    }
}

#[cfg(test)]
#[test]
fn test_new() {
    let m = Vector::new(vec![1u32], vec![1i8]).unwrap();

    let expected_tensor = Tensor::new(vec![1u32], vec![1i8]).unwrap();
    assert_eq!(m.tensor, expected_tensor);
}
