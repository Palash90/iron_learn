//! # Tensor Module - Linear Algebra Core
//!
//! Provides the foundational `Tensor` data structure and comprehensive operations for
//! linear algebra computations essential to machine learning applications.
//!
//! ## Design Philosophy
//!
//! The tensor module is built around these key principles:
//! - **Generic**: Works with all numeric types through the `Numeric` trait
//! - **Safe**: Result-based error handling; no panics on invalid operations
//! - **Flexible**: Both consuming and borrowing variants of operations
//! - **Clear**: Extensive documentation and examples for all public APIs
//!
//! ## Data Representation
//!
//! Tensors are represented in **row-major order** for compatibility with standard
//! mathematical libraries. Shape is defined as a vector of dimensions, supporting
//! multi-dimensional arrays (though currently restricted to 2D matrices).
//!
//! ## Operation Modes
//!
//! Two patterns are provided for each operation:
//! - **Consuming** (`+`, `-`, `*`): Take ownership, suitable for single-use computations
//! - **Borrowing** (`add()`, `sub()`, `mul()`): Borrow references, enable reuse
//!
//! ## Performance Characteristics
//!
//! Operations use foundational algorithms appropriate for educational and small-scale use.
//! GPU acceleration is available through the `gpu_regression` module for large-scale workloads.

use crate::tensor::Tensor;
mod display;
// mod tensor_ops;

use crate::numeric::{Numeric, SignedNumeric};
use std::ops::{Add, Mul, Neg, Sub};

/// The `Tensor` structure is the cornerstone of this library, providing a comprehensive suite of mathematical operations
/// for the manipulation of multidimensional data.
///
/// # Characteristics
///
/// - **Generic**: Type-agnostic through the `Numeric` trait; works with f64, f32, i32, i64, u32, u64, and Complex
/// - **Row-Major Layout**: Data stored in row-major order for standard matrix conventions
/// - **Type Safe**: All operations return `Result` types for explicit error handling
/// - **Flexible API**: Both ownership-taking operators and borrowing methods available
///
/// # Limitations
///
/// Currently restricted to 2-dimensional matrices. N-dimensional support is planned for future releases.
///
/// # Examples
///
/// ```rust
/// # use iron_learn::CpuTensor;
/// // Create a 2x2 matrix
/// let a = CpuTensor::new(vec![2, 2], vec![1, 2, 3, 4]).unwrap();
/// let b = CpuTensor::new(vec![2, 2], vec![5, 6, 7, 8]).unwrap();
///
/// // Borrowing method (non-consuming)
/// let c = a.add(&b).unwrap();
///
/// // Both a and b can still be used
/// let d = a.mul(&b).unwrap();
/// ```

#[derive(Debug, PartialEq, Clone)]
pub struct CpuTensor<T: Numeric> {
    shape: Vec<u32>,
    data: Vec<T>,
}
impl<T> CpuTensor<T>
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

enum OpType {
    EXP = 0,
    SIN = 2,
    COS = 3,
    TAN = 4,
    TANH = 5,
    SIGMOID = 6,
    LOG = 7,
    LN = 8,
}

// This is the actual implementation of all the operations. This is here to avoid the documentation comment clutter.
impl<T: Numeric> CpuTensor<T> {
    fn _sigmoid(t: f64) -> f64 {
        f64::exp(t) / (1.0 + f64::exp(t))
    }

    fn element_op(operand: &Self, op_type: OpType) -> CpuTensor<f64> {
        let result = match op_type {
            OpType::EXP => operand.data.iter().map(|t| f64::exp(t.f64())).collect(),
            OpType::COS => operand.data.iter().map(|t| f64::cos(t.f64())).collect(),
            OpType::SIN => operand.data.iter().map(|t| f64::sin(t.f64())).collect(),
            OpType::TAN => operand.data.iter().map(|t| f64::tan(t.f64())).collect(),
            OpType::TANH => operand.data.iter().map(|t| f64::tanh(t.f64())).collect(),
            OpType::SIGMOID => operand
                .data
                .iter()
                .map(|t| Self::_sigmoid(t.f64()))
                .collect(),
            OpType::LOG => operand.data.iter().map(|t| f64::log10(t.f64())).collect(),
            OpType::LN => operand.data.iter().map(|t| f64::ln(t.f64())).collect(),
        };

        CpuTensor::new(operand.shape.clone(), result).unwrap()
    }

    fn _t(&self) -> Result<Self, String> {
        if self.shape.len() > 2 {
            return Err("Only upto 2D tensors can be transposed.".to_string());
        }

        if self.shape.len() == 1 {
            return Ok(self.clone());
        }

        let new_shape = vec![self.shape[1], self.shape[0]];
        let mut new_data = vec![T::zero(); self.data.len()];
        let (rows, cols) = (self.shape[0] as usize, self.shape[1] as usize);
        for i in 0..rows {
            for j in 0..cols {
                new_data[j * rows + i] = self.data[i * cols + j];
            }
        }
        Ok(Self {
            shape: new_shape,
            data: new_data,
        })
    }
    fn _data(&self) -> Vec<T> {
        self.data.clone()
    }

    fn _shape(&self) -> Vec<u32> {
        self.shape.clone()
    }

    fn _add(&self, rhs: &Self, sub: bool) -> Result<Self, String> {
        let mut result = Vec::with_capacity(self.data.len());

        if self.shape != rhs.shape {
            return Err(format!("ShapeMismatch:The dimensions of two matrices are not compatible for addition/subtraction- {:?} {:?}", self.shape, rhs.shape));
        }

        for i in 0..self.data.len() {
            if sub {
                result.push(self.data[i] - rhs.data[i])
            } else {
                result.push(self.data[i] + rhs.data[i])
            }
        }
        Ok(Self {
            shape: self.shape.clone(),
            data: result,
        })
    }

    fn check_shape(shape: &[u32]) -> Option<Result<CpuTensor<T>, String>> {
        if shape.is_empty() {
            return Some(Err(
                "ShapeError: Tensor must have at least one dimension.".to_string()
            ));
        }

        if shape.len() > 2 {
            return Some(Err(
                "TemporaryShapeRestriction: Currently only accepting tensors upto 2 dimensions"
                    .to_string(),
            ));
        }
        None
    }

    fn calculate_length(shape: &Vec<u32>) -> u32 {
        let mut size = 1;

        for i in shape {
            size *= i;
        }
        size
    }

    fn _new(shape: Vec<u32>, data: Vec<T>) -> Result<Self, String> {
        if let Some(value) = Self::check_shape(&shape) {
            return value;
        }

        let size = Self::calculate_length(&shape);

        if size != (data.len() as u32) {
            let err = format!("DataError: Data length ({}) does not match total num of elements provided by dimensions ({}))", data.len(), size);
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

    fn _cpu_mul(&self, rhs: &Self) -> Result<Self, String> {
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

        Ok(CpuTensor {
            shape: vec![rows as u32, cols as u32],
            data,
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

        self._cpu_mul(rhs)
    }

    fn _s(&self, scalar: T) -> Self {
        let mut new_data = Vec::<T>::new();

        for i in self._data() {
            new_data.push(i * scalar);
        }

        Self {
            shape: self.shape.clone(),
            data: new_data,
        }
    }
}

impl<T: Numeric> Add for CpuTensor<T> {
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
    /// - `Err(String)`: An error message if the shapes of the two tensors do not match.
    ///
    /// # Errors
    /// - `ShapeMismatch`: Returned when the shapes of the two tensors are not identical, preventing element-wise addition.
    ///
    /// # Example
    ///
    /// ```rust
    /// use iron_learn::CpuTensor;
    ///
    /// let tensor_a = CpuTensor::new(vec![2, 2], vec![1, 2, 3, 4]).unwrap();
    /// let tensor_b = CpuTensor::new(vec![2, 2], vec![5, 6, 7, 8]).unwrap();
    /// let tensor_sum = (tensor_a + tensor_b).expect("ShapeMismatch: Mismatch in shape of two Tensors.");
    /// // tensor_sum now contains the element-wise sum of tensor_a and tensor_b
    /// ```
    ///
    /// Note: This method supports all numeric types defined in the `numeric` module, allowing for a wide range of tensor operations.
    fn add(self, rhs: Self) -> Result<Self, String> {
        self._add(&rhs, false)
    }
}

impl<T: Numeric> Sub for CpuTensor<T> {
    type Output = Result<Self, String>;

    /// Implements the subtraction of two `Tensor` instances.
    ///
    /// This implementation of the `Sub` trait enables the subtraction of two tensors with the same shape. The operation is element-wise, meaning each element in one tensor is subtracted to the corresponding element in the other tensor.
    ///
    /// # Type Parameters
    /// - `T`: A type that implements the `Numeric` trait, ensuring the elements can be subtracted together.
    ///
    /// # Output
    /// The output is a `Result` type that either contains:
    /// - `Ok(Tensor)`: A new `Tensor` representing the difference of the two tensors.
    /// - `Err(String)`: An error message if the shapes of the two tensors do not match.
    ///
    /// # Errors
    /// - `ShapeMismatch`: Returned when the shapes of the two tensors are not identical, preventing element-wise addition.
    ///
    /// # Example
    ///
    /// ```rust
    /// use iron_learn::CpuTensor;
    ///
    /// let tensor_a = CpuTensor::new(vec![2, 2], vec![1, 2, 3, 4]).unwrap();
    /// let tensor_b = CpuTensor::new(vec![2, 2], vec![5, 6, 7, 8]).unwrap();
    /// let tensor_sum = (tensor_a - tensor_b).expect("ShapeMismatch: Mismatch in shape of two Tensors.");
    /// // tensor_sum now contains the element-wise sum of tensor_a and tensor_b
    /// ```
    ///
    /// Note: This method supports all numeric types defined in the `numeric` module, allowing for a wide range of tensor operations.
    fn sub(self, rhs: Self) -> Result<Self, String> {
        self._add(&rhs, true)
    }
}

impl<T: Numeric> Mul for CpuTensor<T> {
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
    /// use iron_learn::CpuTensor;
    ///
    /// let tensor_a = CpuTensor::new(vec![2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();
    /// let tensor_b = CpuTensor::new(vec![3, 2], vec![7, 8, 9, 10, 11, 12]).unwrap();
    /// let tensor_product = (tensor_a * tensor_b).expect("ShapeMismatch: The dimensions of two matrices are not matching.");
    /// // tensor_product now contains the product of tensor_a and tensor_b
    /// ```
    ///
    /// Note: This method is designed to support all numeric types defined in the `numeric` module, ensuring compatibility with a wide range of data types.
    fn mul(self, rhs: Self) -> Result<Self, String> {
        self._mul(&rhs)
    }
}

impl<T: SignedNumeric> Neg for CpuTensor<T> {
    type Output = Self;
    fn neg(self) -> Self {
        let result: Vec<T> = self.data.iter().map(|t| -*t).collect();
        CpuTensor::_new(self.shape, result).unwrap()
    }
}


// This trait implementation wires the public API to the internal logic.
impl<T: Numeric> Tensor<T> for CpuTensor<T> where CpuTensor<T>: From<CpuTensor<f64>>{
    /* Creation */
    /// Returns an Empty Tensor with the provided shape, filled with the zero value for T.
    fn empty(shape: &Vec<u32>) -> Self {
        let size = Self::calculate_length(shape) as usize;
        let data = vec![T::zero(); size];
        Self {
            shape: shape.clone(),
            data,
        }
    }

    /// Creates a new tensor with the provided shape and data.
    /// Wires to the internal validation logic.
    fn new(shape: Vec<u32>, data: Vec<T>) -> Result<Self, String> {
        Self::_new(shape, data)
    }

    /* Retrieval */
    /// Returns the shape of the Tensor.
    fn get_shape(&self) -> &Vec<u32> {
        &self.shape
    }

    /// Returns the data (cloning it to give ownership to the caller).
    fn get_data(&self) -> Vec<T> {
        self._data()
    }

    /* Device API */
    /// Synchronize with GPU. As this is a CpuTensor, this is a no-op.
    fn synchronize(&self) {
        // No-op for CPU implementation
    }

    /* Matrix related operations */
    /// Adds two Tensors element-wise.
    /// Wires to the internal addition logic.
    fn add(&self, rhs: &Self) -> Result<Self, String> {
        self._add(rhs, false)
    }

    /// Subtracts rhs from current tensor element-wise.
    /// Wires to the internal subtraction logic.
    fn sub(&self, rhs: &Self) -> Result<Self, String> {
        self._add(rhs, true)
    }

    /// Matrix multiplication.
    /// Wires to the internal multiplication logic.
    fn mul(&self, rhs: &Self) -> Result<Self, String> {
        // The implementation _mul performs matrix multiplication
        self._mul(rhs)
    }

    /// Transpose a 2D matrix.
    /// Wires to the internal transpose logic.
    fn t(&self) -> Result<Self, String> {
        self._t()
    }

    /// Hadamard product (element-wise multiplication).
    /// Wires to the internal Hadamard product logic.
    fn multiply(&self, rhs: &Self) -> Result<Self, String> {
        self.hadamard(rhs)
    }

    /* Mathematical functions */
    // Note: All element-wise mathematical functions use f64 for computation
    // as per your existing `element_op` helper.
    // The resulting tensor will contain f64 data.

    /// Element wise sigmoid function.
    fn sigmoid(&self) -> Result<Self, String> {
        Ok(Self::element_op(self, OpType::SIGMOID).into()) // .into() casts CpuTensor<f64> to Self
    }

    /// Element wise log base 10 implementation.
    fn log(&self) -> Result<Self, String> {
        Ok(Self::element_op(self, OpType::LOG).into())
    }

    /// Element wise natural log.
    fn ln(&self) -> Result<Self, String> {
        Ok(Self::element_op(self, OpType::LN).into())
    }

    /// Element wise sin.
    fn sin(&self) -> Result<Self, String> {
        Ok(Self::element_op(self, OpType::SIN).into())
    }

    /// Element wise cos.
    fn cos(&self) -> Result<Self, String> {
        Ok(Self::element_op(self, OpType::COS).into())
    }

    /// Element wise tan.
    fn tan(&self) -> Result<Self, String> {
        Ok(Self::element_op(self, OpType::TAN).into())
    }

    /// Element wise tanh.
    fn tanh(&self) -> Result<Self, String> {
        Ok(Self::element_op(self, OpType::TANH).into())
    }

    /// Element wise exponentiation.
    fn exp(&self) -> Result<Self, String> {
        Ok(Self::element_op(self, OpType::EXP).into())
    }

    /// Element wise scaling (multiplication by a scalar).
    fn scale(&self, scalar: T) -> Result<Self, String> {
        Ok(self._s(scalar))
    }

    /* Reducers */
    /// Reduces rows column wise (sums all elements).
    /// **Note:** Based on your function name `sum`, I am implementing a full sum reduction (summing all elements into a 1-element tensor). If you intended a column or row-wise reduction that keeps one dimension, the logic will need adjustment.
    fn sum(&self) -> Result<Self, String> {
        let sum_val = self.data.iter().fold(T::zero(), |acc, &x| acc + x);
        
        // Return a tensor with a single element and shape [1]
        Self::new(vec![1], vec![sum_val])
    }
}

#[cfg(test)]
#[test]
fn test_new() {
    let t = CpuTensor::new(vec![1u32, 2u32], vec![1i8, 2i8]).unwrap();

    assert_eq!(t.shape, vec![1u32, 2u32]);
    assert_eq!(t.data, vec![1i8, 2i8]);
}
