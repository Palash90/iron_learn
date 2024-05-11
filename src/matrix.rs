//! The `matrix` module acts as a wrapper over the `tensor` module, providing a specialized `Matrix` type
//! for matrix-specific operations. This abstraction simplifies the usage of tensors when working with
//! matrix data structures in machine learning applications.

use crate::numeric::Numeric;
use crate::tensor::Tensor;
use std::ops::Add;
use std::ops::Mul;

#[derive(Debug, PartialEq)]
pub struct Matrix<T: Numeric> {
    tensor: Tensor<T>,
}

impl<T: Numeric> Matrix<T> {
    /// Constructs a new `Matrix`.
    ///
    /// This function initializes a `Matrix` with a specified shape and data. It validates that the shape provided has exactly two dimensions, as required for matrix structures.
    ///
    /// # Parameters
    /// - `shape`: A `Vec<u32>` representing the dimensions of the matrix (rows and columns).
    /// - `data`: A `Vec<T>` containing the matrix's elements, where `T` is a numeric type.
    ///
    /// # Returns
    /// A `Result` which is either:
    /// - `Ok(Matrix)`: A new instance of `Matrix` if the shape is valid.
    /// - `Err(String)`: An error message if the provided shape does not have exactly two dimensions.
    ///
    /// # Errors
    /// - `MatrixShapeError`: Returned when the provided shape vector does not represent a two-dimensional matrix.
    ///
    /// # Example
    ///
    /// ```rust
    /// use iron_learn::matrix::Matrix;
    ///
    /// let matrix = Matrix::new(vec![2, 3], vec![1, 2, 3, 4, 5, 6])
    ///     .expect("MatrixShapeError: Matrix can have only two dimensions.");
    /// // matrix is now a 2x3 matrix with the provided data
    /// ```
    ///
    /// Note: The `new` function ensures that the `Matrix` type is compatible with all numeric types defined in the `numeric` module.
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
    /// Performs the Hadamard product (element-wise multiplication) of two matrices.
    ///
    /// This method is a convenient wrapper over the `Tensor` Hadamard product, allowing for the element-wise multiplication of two matrices. It leverages the underlying tensor multiplication to compute the result.
    ///
    /// # Parameters
    /// - `rhs`: The right-hand side matrix for the multiplication.
    ///
    /// # Returns
    /// A `Result` which is either:
    /// - `Ok(Matrix)`: A new `Matrix` instance representing the Hadamard product of the two matrices.
    /// - `Err(&'static str)`: An error message if the underlying tensors' shapes do not allow for element-wise multiplication.
    ///
    /// # Example
    ///
    /// ```rust
    /// use iron_learn::matrix::Matrix;
    ///
    /// let matrix_a = Matrix::new(vec![2, 2], vec![1, 2, 3, 4]).expect("Matrix initialization failed.");
    /// let matrix_b = Matrix::new(vec![2, 2], vec![5, 6, 7, 8]).expect("Matrix initialization failed.");
    /// let matrix_product = matrix_a.multiply(matrix_b).expect("Hadamard product failed due to shape mismatch.");
    /// // matrix_product now contains the Hadamard product of matrix_a and matrix_b
    /// ```
    pub fn multiply(self, rhs: Self) -> Result<Self, &'static str> {
        Ok(Self {
            tensor: self.tensor.multiply(rhs.tensor)?,
        })
    }
}

impl<T: Numeric> Add for Matrix<T> {
    type Output = Result<Self, &'static str>;
    /// Implements the addition of two `Matrix` instances.
    ///
    /// This method provides the functionality to add two matrices together. It utilizes the underlying tensor addition to compute the sum of the matrices, ensuring that the operation is performed element-wise.
    ///
    /// # Parameters
    /// - `rhs`: The right-hand side matrix for the addition.
    ///
    /// # Returns
    /// A `Result` which is either:
    /// - `Ok(Matrix)`: A new `Matrix` instance representing the sum of the two matrices.
    /// - `Err(&'static str)`: An error message if the underlying tensors' shapes do not allow for addition.
    ///
    /// # Example
    ///
    /// ```rust
    /// use iron_learn::Matrix;
    ///
    /// let matrix_a = Matrix::new(vec![2, 2], vec![1, 2, 3, 4]).expect("Matrix initialization failed.");
    /// let matrix_b = Matrix::new(vec![2, 2], vec![5, 6, 7, 8]).expect("Matrix initialization failed.");
    /// let matrix_sum = (matrix_a + matrix_b).expect("Addition failed due to shape mismatch.");
    /// // matrix_sum now contains the sum of matrix_a and matrix_b
    /// ```
    fn add(self, rhs: Self) -> Result<Self, &'static str> {
        let result = self.tensor + rhs.tensor;
        Ok(Self { tensor: result? })
    }
}

impl<T: Numeric> Mul for Matrix<T> {
    type Output = Result<Self, String>;
    /// Implements the multiplication of two `Matrix` instances.
    ///
    /// This method enables the multiplication of two matrices, akin to traditional matrix multiplication. It relies on the underlying tensor multiplication to calculate the product, adhering to the rules of linear algebra.
    ///
    /// # Parameters
    /// - `rhs`: The right-hand side matrix for the multiplication.
    ///
    /// # Returns
    /// A `Result` which is either:
    /// - `Ok(Matrix)`: A new `Matrix` instance representing the product of the two matrices.
    /// - `Err(String)`: A formatted error message if the dimensions of the two tensors do not allow for multiplication.
    ///
    /// # Example
    ///
    /// ```rust
    /// use iron_learn::Matrix;
    ///
    /// let matrix_a = Matrix::new(vec![2, 3], vec![1, 2, 3, 4, 5, 6]).expect("Matrix initialization failed.");
    /// let matrix_b = Matrix::new(vec![3, 2], vec![7, 8, 9, 10, 11, 12]).expect("Matrix initialization failed.");
    /// let matrix_product = (matrix_a * matrix_b).expect("Multiplication failed due to dimension mismatch.");
    /// // matrix_product now contains the product of matrix_a and matrix_b
    /// ```
    ///
    /// Note: This method assumes that the number of columns in the first matrix matches the number of rows in the second matrix, as required by matrix multiplication rules.

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
