//! The `tensor` module provides the foundational structures and operations
//! for linear algebra computations integral to machine learning applications.

mod display;
use crate::numeric::Numeric;
use std::ops::{Add, Mul};

/// The `Tensor` structure is the cornerstone of this library, providing a comprehensive suite of mathematical operations
/// for the manipulation of multidimensional data. It is designed to be compatible with all numeric types defined in the `numeric` module,
/// ensuring versatility and broad applicability in various machine learning contexts.

#[derive(Debug, PartialEq)]
pub struct Tensor<T: Numeric> {
    shape: Vec<u32>,
    data: Vec<T>,
}

// This is the actual implementation of all the operations. This is here to avoid the documentation comment clutter.
impl<T: Numeric> Tensor<T> {
    fn _data(&self) -> Vec<T> {
        self.data.clone()
    }

    fn _add(&self, rhs: &Self) -> Result<Self, String> {
        let mut result = Vec::with_capacity(self.data.len());

        if self.shape != rhs.shape {
            return Err(format!("ShapeMismatch:The dimensions of two matrices are not compatible for addition- {:?} {:?}", self.shape, rhs.shape));
        }

        for i in 0..self.data.len() {
            result.push(self.data[i] + rhs.data[i])
        }

        Ok(Self {
            shape: self.shape.clone(),
            data: result,
        })
    }

    fn _new(shape: Vec<u32>, data: Vec<T>) -> Result<Self, String> {
        if shape.len() == 0 {
            return Err(format!(
                "ShapeError: Tensor must have at least one dimension."
            ));
        }

        if shape.len() > 2 {
            return Err(format!(
                "TemporaryShapeRestriction: Currently only accepting tensors upto 2 dimensions"
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

    fn hadamard(&self, rhs: &Self) -> Result<Self, String> {
        let mut result = Vec::with_capacity(self.data.len());

        if self.shape != rhs.shape {
            return Err(format!(
                "ShapeMismatch: Mismatch in shape of two Tensors {:?} {:?}",
                self.shape, rhs.shape
            ));
        }

        for i in 0..self.data.len() {
            result.push(self.data[i] * rhs.data[i])
        }

        Ok(Self {
            shape: self.shape.clone(),
            data: result,
        })
    }

    fn _mul(&self, rhs: &Self) -> Result<Self, String> {
        if self.shape[1] != rhs.shape[0] {
            let s = format!(
                "ShapeMismatch:The dimensions of two matrices are not compatible for multiplication- {:?} {:?}",
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

// The public API of Tensor type
impl<T: Numeric> Tensor<T> {
    /// Constructs a new instance of a `Tensor`.
    ///
    /// This associated function initializes a `Tensor` with a specified shape and data. It requires two parameters:
    /// - `shape`: A `Vec<u32>` that defines the dimensions of the `Tensor`. The length is treated the number of dimensions, while each item in the Vec is considered number of elements in corresponding dimension.
    /// - `data`: A `Vec<T>` containing the elements of the `Tensor`, where `T` is a numeric type defined in the `numeric` module.
    ///
    /// # Returns
    /// - When the operation is succeeded, the function returns a new `Tensor` object
    /// - In case of any failure, the function return an owned String.
    ///
    /// # Example
    ///
    /// ```rust
    /// use iron_learn::Tensor;
    ///
    /// let tensor = Tensor::new(vec![2, 2], vec![1, 2, 3, 4]); // Initializes a 2x2 tensor
    /// ```
    ///
    ///
    /// # Errors
    /// - **ShapeError**: If the shape vector length is passed as 0
    /// - **TemporaryShapeRestriction**: Currently, tensors are limited to a maximum of two dimensions. This restriction will be lifted as further methods are implemented.
    /// - **DataError**: This error occurs if the total number of elements in the `data` vector does not correspond to the product of the `shape` dimensions.
    ///
    pub fn new(shape: Vec<u32>, data: Vec<T>) -> Result<Self, String> {
        Self::_new(shape, data)
    }

    /// Retrieves the underlying data from a `Tensor`.
    ///
    /// This method returns the data held within the `Tensor` as a vector of type `T`, where `T` encompasses all numeric types supported by the library.
    /// It is a fundamental method that allows for direct access to the tensor's data for further processing or analysis.
    ///
    /// # Returns
    /// A `Vec<T>` containing the tensor's data.
    ///
    /// # Example
    ///
    /// ```rust
    /// use iron_learn::Tensor;
    ///
    /// let tensor = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    /// let data = tensor.get_data(); // Retrieves the data as Vec<f64>
    ///
    /// assert_eq!(vec![1.0, 2.0, 3.0, 4.0], data);
    /// ```
    ///
    /// Note: The `get_data` function is designed to work seamlessly with all numeric types defined in the `numeric` module, ensuring broad compatibility.
    pub fn get_data(&self) -> Vec<T> {
        self._data()
    }

    /// Performs the Hadamard product (element-wise multiplication) on two tensors.
    ///
    /// This method calculates the element-wise product of the invoking tensor with another tensor of the same shape. The operation is also known as the Hadamard product, which is a critical operation in various machine learning algorithms.
    ///
    /// # Parameters
    /// - `rhs`: The right-hand side tensor to multiply with.
    ///
    /// # Returns
    /// A `Result` which is either:
    /// - `Ok(Tensor)`: A new `Tensor` representing the Hadamard product of the two tensors.
    /// - `Err(&'static str)`: An error message if the shapes of the two tensors do not match.
    ///
    /// # Errors
    /// - `ShapeMismatch`: Returned when the shapes of the two tensors are not identical.
    ///
    /// # Example
    ///
    /// ```rust
    /// use iron_learn::Tensor;
    ///
    /// let tensor_a = Tensor::new(vec![2, 2], vec![1, 2, 3, 4]).unwrap();
    /// let tensor_b = Tensor::new(vec![2, 2], vec![5, 6, 7, 8]).unwrap();
    /// let tensor_product = tensor_a.multiply(&tensor_b).expect("ShapeMismatch: Mismatch in shape of two Tensors.");
    /// // tensor_product now contains the element-wise product of tensor_a and tensor_b
    /// ```
    ///
    /// Note: This method supports all numeric types defined in the `numeric` module, ensuring compatibility with a wide range of data types.
    pub fn multiply(&self, rhs: &Self) -> Result<Self, String> {
        self.hadamard(&rhs)
    }
}

impl<T: Numeric> Add for Tensor<T> {
    type Output = Result<Self, String>;

    /// Implements the addition of two `Tensor` instances.
    ///
    /// This implementation of the `Add` trait enables the addition of two tensors with the same shape. The operation is element-wise, meaning each element in one tensor is added to the corresponding element in the other tensor.
    ///
    /// # Type Parameters
    /// - `T`: A type that implements the `Numeric` trait, ensuring the elements can be added together.
    ///
    /// # Output
    /// The output is a `Result` type that either contains:
    /// - `Ok(Tensor)`: A new `Tensor` representing the sum of the two tensors.
    /// - `Err(&'static str)`: An error message if the shapes of the two tensors do not match.
    ///
    /// # Errors
    /// - `ShapeMismatch`: Returned when the shapes of the two tensors are not identical, preventing element-wise addition.
    ///
    /// # Example
    ///
    /// ```rust
    /// use iron_learn::Tensor;
    ///
    /// let tensor_a = Tensor::new(vec![2, 2], vec![1, 2, 3, 4]).unwrap();
    /// let tensor_b = Tensor::new(vec![2, 2], vec![5, 6, 7, 8]).unwrap();
    /// let tensor_sum = (tensor_a + tensor_b).expect("ShapeMismatch: Mismatch in shape of two Tensors.");
    /// // tensor_sum now contains the element-wise sum of tensor_a and tensor_b
    /// ```
    ///
    /// Note: This method supports all numeric types defined in the `numeric` module, allowing for a wide range of tensor operations.
    fn add(self, rhs: Self) -> Result<Self, String> {
        self._add(&rhs)
    }
}

impl<T: Numeric> Mul for Tensor<T> {
    type Output = Result<Self, String>;

    /// Implements tensor multiplication using the `*` operator.
    ///
    /// This method enables the multiplication of two tensors, akin to matrix multiplication. It requires that the number of columns in the first tensor matches the number of rows in the second tensor, following the rules of matrix multiplication.
    ///
    /// # Type Parameters
    /// - `T`: A type that implements the `Numeric` trait, which includes all numeric types supported by the library.
    ///
    /// # Output
    /// The output is a `Result` type that either contains:
    /// - `Ok(Tensor)`: A new `Tensor` representing the product of the two tensors.
    /// - `Err(String)`: A formatted error message if the dimensions of the two tensors do not allow for multiplication.
    ///
    /// # Errors
    /// - `ShapeMismatch`: Returned when the number of columns in the first tensor does not match the number of rows in the second tensor.
    ///
    /// # Example
    ///
    /// ```rust
    /// use iron_learn::Tensor;
    ///
    /// let tensor_a = Tensor::new(vec![2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();
    /// let tensor_b = Tensor::new(vec![3, 2], vec![7, 8, 9, 10, 11, 12]).unwrap();
    /// let tensor_product = (tensor_a * tensor_b).expect("ShapeMismatch: The dimensions of two matrices are not matching.");
    /// // tensor_product now contains the product of tensor_a and tensor_b
    /// ```
    ///
    /// Note: This method is designed to support all numeric types defined in the `numeric` module, ensuring compatibility with a wide range of data types.
    fn mul(self, rhs: Self) -> Result<Self, String> {
        self._mul(&rhs)
    }
}

#[cfg(test)]
#[test]
fn test_new() {
    let t = Tensor::new(vec![1u32, 2u32], vec![1i8, 2i8]).unwrap();

    assert_eq!(t.shape, vec![1u32, 2u32]);
    assert_eq!(t.data, vec![1i8, 2i8]);
}
