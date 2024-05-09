use crate::numeric::Numeric;
use crate::tensor::Tensor;
use std::ops::Add;

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
}

impl<T: Numeric> Add for Matrix<T> {
    type Output = Result<Self, &'static str>;

    fn add(self, rhs: Self) -> Result<Self, &'static str> {
        let result = self.tensor + rhs.tensor;
        Ok(Self { tensor: result? })
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_new() {
        let m = Matrix::new(vec![1u32, 2u32], vec![1i8, 2i8]).unwrap();

        let expected_tensor = Tensor::new(vec![1u32, 2u32], vec![1i8, 2i8]).unwrap();
        assert_eq!(m.tensor, expected_tensor);
    }

    #[test]
    #[should_panic(expected = "MatrixShapeError")]
    fn test_new_panic_on_shape() {
        Matrix::new(vec![1u32, 2u32, 3u32], vec![1, 2, 3]).unwrap();
    }

    #[test]
    fn test_add() {
        let m1 = Matrix::new(vec![1u32, 2u32], vec![1, 2]).unwrap();
        let m2 = Matrix::new(vec![1u32, 2u32], vec![3, 4]).unwrap();

        let expected = Matrix::new(vec![1u32, 2u32], vec![4, 6]).unwrap();
        let actual = (m1 + m2).unwrap();
        assert_eq!(expected, actual);
    }
}
