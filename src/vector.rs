//! The `vector` module acts as a wrapper over the `tensor` module, providing a specialized `Vector` type
//! for matrix-specific operations. This abstraction simplifies the usage of vectors when working
//! in machine learning applications.

use crate::numeric::Numeric;
use crate::tensor::Tensor;
use std::ops::Add;
use std::ops::Mul;

/// A `Vector` struct that encapsulates a `Tensor` for one-dimensional numerical data.
///
/// This struct is a specialized version of a `Tensor` that is constrained to one dimension, representing a mathematical vector. It inherits all numerical operations from the `Tensor` through the `Numeric` trait.
#[derive(Debug, PartialEq)]
pub struct Vector<T: Numeric> {
    tensor: Tensor<T>,
}

impl<T: Numeric> Vector<T> {
    /// Constructs a new `Vector`.
    ///
    /// This function initializes a `Vector` with a specified shape and data. It validates that the shape provided has exactly one dimension, as required for vector structures.
    ///
    /// # Parameters
    /// - `shape`: A `Vec<u32>` representing the dimension of the vector.
    /// - `data`: A `Vec<T>` containing the vector's elements, where `T` is a numeric type.
    ///
    /// # Returns
    /// A `Result` which is either:
    /// - `Ok(Vector)`: A new instance of `Vector` if the shape is valid.
    /// - `Err(String)`: An error message if the provided shape does not have exactly one dimension.
    ///
    /// # Errors
    /// - `VectorShapeError`: Returned when the provided shape vector does not represent a one-dimensional vector.
    ///
    /// # Example
    ///
    /// ```rust
    /// use iron_learn::Vector;
    ///
    /// let vector = Vector::new(vec![3], vec![1, 2, 3])
    ///     .expect("VectorShapeError: Vector can have only one dimension.");
    /// // vector is now a 1x3 vector with the provided data
    /// ```
    ///
    /// Note: The `new` function ensures that the `Vector` type is compatible with all numeric types defined in the `numeric` module.
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

    /// Retrieves the underlying data from a `Matrix`.
    ///
    /// # Returns
    /// A `Vec<T>` containing the tensor's data.
    ///
    /// # Example
    ///
    /// ```rust
    /// use iron_learn::Matrix;
    ///
    /// let m = Matrix::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    /// let data = m.get_data(); // Retrieves the data as Vec<f64>
    ///
    /// assert_eq!(vec![1.0, 2.0, 3.0, 4.0], data);
    /// ```
    ///
    /// Note: The `get_data` function is designed to work seamlessly with all numeric types defined in the `numeric` module, ensuring broad compatibility.
    pub fn get_data(self) -> Vec<T> {
        self.tensor.get_data()
    }
}

impl<T: Numeric> Add for Vector<T> {
    type Output = Result<Self, String>;
    /// Implements the addition of two `Vector` instances.
    ///
    /// This method enables the addition of two vectors by leveraging the addition operation defined for tensors. It ensures that the operation is performed element-wise, adhering to vector addition rules.
    ///
    /// # Parameters
    /// - `rhs`: The right-hand side `Vector` to be added.
    ///
    /// # Returns
    /// A `Result` which is either:
    /// - `Ok(Vector)`: A new `Vector` instance representing the sum of the two vectors.
    /// - `Err(&'static str)`: An error message if the underlying tensors' shapes do not allow for addition.
    ///
    /// # Example
    ///
    /// ```rust
    /// use iron_learn::Vector;
    ///
    /// let vector_a = Vector::new(vec![3], vec![1, 2, 3]).expect("Vector initialization failed.");
    /// let vector_b = Vector::new(vec![3], vec![4, 5, 6]).expect("Vector initialization failed.");
    /// let vector_sum = (vector_a + vector_b).expect("Addition failed due to shape mismatch.");
    /// // vector_sum now contains the sum of vector_a and vector_b
    /// ```
    fn add(self, rhs: Self) -> Result<Self, String> {
        let result = self.tensor + rhs.tensor;
        Ok(Self { tensor: result? })
    }
}

impl<T: Numeric> Mul for Vector<T> {
    type Output = Result<T, String>;
    /// Implements the dot product of two `Vector` instances.
    ///
    /// This method calculates the dot product of two vectors, which is a fundamental operation in many machine learning algorithms. The dot product is computed by performing element-wise multiplication followed by a summation of the resulting products.
    ///
    /// # Parameters
    /// - `rhs`: The right-hand side `Vector` for the dot product.
    ///
    /// # Returns
    /// A `Result` which is either:
    /// - `Ok(T)`: The result of the dot product as a single numeric value of type `T`.
    /// - `Err(String)`: A formatted error message if the underlying tensors' shapes do not allow for the dot product.
    ///
    /// # Example
    ///
    /// ```rust
    /// use iron_learn::Vector;
    ///
    /// let vector_a = Vector::new(vec![3], vec![1, 2, 3]).expect("Vector initialization failed.");
    /// let vector_b = Vector::new(vec![3], vec![4, 5, 6]).expect("Vector initialization failed.");
    /// let dot_product = (vector_a * vector_b).expect("Dot product failed due to shape mismatch.");
    /// // dot_product now contains the result of the dot product of vector_a and vector_b
    /// ```
    ///
    /// Note: This method assumes that both vectors are of the same shape and the dot product is defined for them.

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
    let m = Vector::new(vec![1], vec![1i8]).unwrap();

    let expected_tensor = Tensor::new(vec![1], vec![1i8]).unwrap();
    assert_eq!(m.tensor, expected_tensor);
}
