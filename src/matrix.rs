use crate::numeric::Numeric;
use crate::tensor::Tensor;
use std::ops::Add;
use std::ops::Mul;

#[derive(Debug, PartialEq)]
pub struct Matrix<T: Numeric> {
    tensor: Tensor<T>,
}

impl<T: Numeric> Matrix<T> {
    pub fn new(shape: Vec<u32>, data: Vec<T>) -> Result<Self, String> {
        if shape.len() != 2 {
            return Err(format!(
                "MatrixShapeError: Matrix can have only two dimensions, provided {}",
                shape.len()
            ));
        }

        let t = Tensor::new(shape, data);

        Ok(Self { tensor: t? })
    }

    pub fn multiply(self, rhs: Self) -> Result<Self, &'static str> {
        Ok(Self {
            tensor: self.tensor.multiply(rhs.tensor)?,
        })
    }
}

impl<T: Numeric> Add for Matrix<T> {
    type Output = Result<Self, &'static str>;

    fn add(self, rhs: Self) -> Result<Self, &'static str> {
        let result = self.tensor + rhs.tensor;
        Ok(Self { tensor: result? })
    }
}

impl<T: Numeric> Mul for Matrix<T> {
    type Output = Result<Self, String>;

    fn mul(self, rhs: Self) -> Result<Self, String> {
        let result = self.tensor * rhs.tensor;
        Ok(Self { tensor: result? })
    }
}

#[cfg(test)]
#[test]
fn test_new() {
    let m = Matrix::new(vec![1, 2], vec![1i8, 2i8]).unwrap();

    let expected_tensor = Tensor::new(vec![1, 2], vec![1i8, 2i8]).unwrap();
    assert_eq!(m.tensor, expected_tensor);
}
