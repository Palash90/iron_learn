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

mod display;
mod tensor_ops;

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

struct BroadcastInfo {
    result_shape: Vec<u32>,
    broadcast_self: bool,
    broadcast_rhs: bool,
    is_row_broadcast: bool,
}

impl BroadcastInfo {
    fn is_element_wise(&self) -> bool {
        !self.broadcast_self && !self.broadcast_rhs
    }
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

    fn determine_broadcast_info(
        self_matrix: &Self,
        rhs_matrix: &Self,
    ) -> Result<BroadcastInfo, String> {
        let self_shape = &self_matrix.shape;
        let rhs_shape = &rhs_matrix.shape;

        // 1. Element-wise (Shapes are identical)
        if self_shape == rhs_shape {
            return Ok(BroadcastInfo {
                result_shape: self_shape.clone(),
                broadcast_self: false,
                broadcast_rhs: false,
                is_row_broadcast: false, // Doesn't matter for element-wise
            });
        }

        // 2. Vector-Matrix or Matrix-Vector Addition (One shape has length 1)
        if self_shape.len() == 1 || rhs_shape.len() == 1 {
            // For simplicity, let's only support 1D + 2D for now, as in the original code.
            let (vec, mat) = if self_shape.len() == 1 {
                (self_matrix, rhs_matrix)
            } else if rhs_shape.len() == 1 {
                (rhs_matrix, self_matrix)
            } else {
                // Unhandled shape combination beyond 1D/2D
                return Err(format!(
                "ShapeMismatch: Cannot broadcast vector of shape {:?} with matrix of shape {:?}",
                self_shape, rhs_shape
            ));
            };

            // Assume vec.shape is [vec_len] and mat.shape is [rows, cols]
            let vec_len = vec.shape[0];
            let rows = mat.shape[0];
            let cols = mat.shape[1];

            // Row Broadcast: vector length matches the number of rows
            let is_row_broadcast = vec_len == rows;
            // Col Broadcast: vector length matches the number of columns
            let is_col_broadcast = vec_len == cols;

            let compatible = is_row_broadcast || is_col_broadcast;
            if !compatible {
                return Err(format!("ShapeMismatch: The dimensions of two matrices are not compatible for addition/subtraction- {:?} {:?}", self_shape, rhs_shape));
            }

            let broadcast_self = self_shape.len() == 1;
            let broadcast_rhs = rhs_shape.len() == 1;

            return Ok(BroadcastInfo {
                result_shape: mat.shape.clone(),
                broadcast_self,
                broadcast_rhs,
                is_row_broadcast,
            });
        }

        // 3. Matrix-Matrix Broadcasting (Both shapes are length 2, e.g., 2x3 + 1x3)
        if self_shape.len() == 2 && rhs_shape.len() == 2 {
            let rows_compatible =
                self_shape[0] == rhs_shape[0] || self_shape[0] == 1 || rhs_shape[0] == 1;
            let cols_compatible =
                self_shape[1] == rhs_shape[1] || self_shape[1] == 1 || rhs_shape[1] == 1;

            if !rows_compatible || !cols_compatible {
                return Err(format!("ShapeMismatch: The dimensions of two matrices are not compatible for addition/subtraction- {:?} {:?}", self_shape, rhs_shape));
            }

            let mut result_shape = vec![0; 2];
            result_shape[0] = self_shape[0].max(rhs_shape[0]);
            result_shape[1] = self_shape[1].max(rhs_shape[1]);

            let broadcast_self = self_shape != &result_shape;
            let broadcast_rhs = rhs_shape != &result_shape;

            return Ok(BroadcastInfo {
                result_shape,
                broadcast_self,
                broadcast_rhs,
                is_row_broadcast: false, // Not relevant for this case
            });
        }

        // Catch-all for unsupported dimensions
        Err(format!(
            "ShapeMismatch: Unsupported number of dimensions for addition/subtraction- {:?} {:?}",
            self_shape, rhs_shape
        ))
    }

    fn _add(&self, rhs: &Self, sub: bool) -> Result<Self, String> {
        // 1. Determine broadcasting strategy and result shape
        let info = Self::determine_broadcast_info(self, rhs)?;

        // Setup the operation closure
        let op: fn(T, T) -> T = if sub { |a, b| a - b } else { |a, b| a + b };

        let result_length = info.result_shape.iter().product::<u32>();
        // Use 1 for the dimension if it's 1D, otherwise use the shape value.
        let rows = info.result_shape.get(0).copied().unwrap_or(1);
        let cols = info.result_shape.get(1).copied().unwrap_or(1); 

        let mut result = Vec::with_capacity(result_length as usize);

        // 2. Perform the calculation based on the strategy
        if info.is_element_wise() {
            // Simple element-wise addition/subtraction
            for i in 0..result_length as usize {
                // Casting to usize once here is clearer
                result.push(op(self.data[i], rhs.data[i]));
            }
        } else {
            // Broadcasting required: Loop through the result indices (r, c)
            let result_rows = rows as usize;
            let result_cols = cols as usize;

            for r in 0..result_rows {
                for c in 0..result_cols {
                    // Calculate the index for 'self'
                    let index_self = if info.broadcast_self {
                        // Pass the original matrix's shape for 2D broadcasting logic
                        self.broadcast_index(r, c, self.shape[0] as usize, self.shape[1] as usize, info.is_row_broadcast)
                    } else {
                        r * result_cols + c
                    };

                    // Calculate the index for 'rhs'
                    let index_rhs = if info.broadcast_rhs {
                        // Pass the original matrix's shape for 2D broadcasting logic
                        rhs.broadcast_index(r, c, rhs.shape[0] as usize, rhs.shape[1] as usize, info.is_row_broadcast)
                    } else {
                        r * result_cols + c
                    };

                    result.push(op(self.data[index_self], rhs.data[index_rhs]));
                }
            }
        }

        Ok(Self {
            shape: info.result_shape,
            data: result,
        })
    }

    // Adjusted broadcast_index method for clarity
    fn broadcast_index(
        &self,
        r: usize,
        c: usize,
        // The original logic only used self.shape, so let's simplify the arguments
        // and rely on self.shape being correct.
        _src_rows: usize, // No longer needed
        _src_cols: usize, // No longer needed
        is_row_broadcast: bool,
    ) -> usize {
        match self.shape.len() {
            // Case 1: Vector (1D) broadcasting to a 2D matrix
            1 => {
                let vector_len = self.data.len();
                if is_row_broadcast {
                    // Vector matches matrix rows: vector element is chosen by row index (r)
                    // The vector is broadcast across columns.
                    r % vector_len
                } else {
                    // Vector matches matrix columns: vector element is chosen by column index (c)
                    // The vector is broadcast across rows.
                    c % vector_len
                }
            }
            // Case 2: Matrix (2D) broadcasting (e.g., 1x3 broadcasting to 5x3)
            2 => {
                let src_rows = self.shape[0] as usize;
                let src_cols = self.shape[1] as usize;
                
                // Determine the row index: if source rows is 1 (broadcasted), row index is 0. Otherwise, use result row index r.
                let src_r = if src_rows == 1 { 0 } else { r };
                // Determine the column index: if source cols is 1 (broadcasted), column index is 0. Otherwise, use result column index c.
                let src_c = if src_cols == 1 { 0 } else { c };
                
                // Calculate the flat index in the source data
                src_r * src_cols + src_c
            }
            _ => panic!("Unsupported broadcast dimension in broadcast_index"),
        }
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

// The public API of Tensor type
impl<T: Numeric> CpuTensor<T> {
    /// Return a new instance of a `Tensor` with each value raised to `e`.
    ///
    /// It requires one parameter:
    /// - `operand`: A `Tensor` object.
    ///
    /// # Returns
    /// - New `Tensor` instance with all the values raised to `e` of the input `Tensor`
    ///
    /// # Example
    ///
    /// ```rust
    /// use iron_learn::CpuTensor;
    ///
    /// let tensor = CpuTensor::exp(&CpuTensor::new(vec![2, 2], vec![1, 2, 3, 4]).unwrap());
    /// ```
    ///
    pub fn exp(operand: &Self) -> CpuTensor<f64> {
        Self::element_op(operand, OpType::EXP)
    }

    pub fn sin(operand: &Self) -> CpuTensor<f64> {
        Self::element_op(operand, OpType::SIN)
    }

    pub fn cos(operand: &Self) -> CpuTensor<f64> {
        Self::element_op(operand, OpType::COS)
    }

    pub fn tan(operand: &Self) -> CpuTensor<f64> {
        Self::element_op(operand, OpType::TAN)
    }

    pub fn tanh(operand: &Self) -> CpuTensor<f64> {
        Self::element_op(operand, OpType::TANH)
    }

    pub fn log(operand: &Self) -> CpuTensor<f64> {
        Self::element_op(operand, OpType::LOG)
    }

    pub fn ln(operand: &Self) -> CpuTensor<f64> {
        Self::element_op(operand, OpType::LN)
    }

    pub fn sigmoid(operand: &Self) -> CpuTensor<f64> {
        Self::element_op(operand, OpType::SIGMOID)
    }

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
    /// use iron_learn::CpuTensor;
    ///
    /// let tensor = CpuTensor::new(vec![2, 2], vec![1, 2, 3, 4]); // Initializes a 2x2 tensor
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
    /// This method returns a copy of the data held within the `Tensor` as a vector of type `T`, where `T` encompasses all numeric types supported by the library.
    /// It is a fundamental method that allows for direct access to the tensor's data for further processing or analysis.
    ///
    /// # Returns
    /// A `Vec<T>` containing a copy of the tensor's data.
    ///
    /// # Example
    ///
    /// ```rust
    /// use iron_learn::CpuTensor;
    ///
    /// let tensor = CpuTensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    /// let data = tensor.get_data(); // Retrieves the data as Vec<f64>
    ///
    /// assert_eq!(vec![1.0, 2.0, 3.0, 4.0], data);
    /// ```
    ///
    /// Note: The `get_data` function is designed to work seamlessly with all numeric types defined in the `numeric` module, ensuring broad compatibility.
    pub fn get_data(&self) -> Vec<T> {
        self._data()
    }

    /// Retrieves the shape of the `Tensor`.
    ///
    /// # Returns
    /// A `Vec<u32>` containing a copy of the tensor's shape.
    ///
    /// # Example
    ///
    /// ```rust
    /// use iron_learn::CpuTensor;
    ///
    /// let tensor = CpuTensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    /// let shape = tensor.get_shape(); // Retrieves the data as Vec<f64>
    ///
    /// assert_eq!(vec![2, 2], shape);
    /// ```
    ///
    pub fn get_shape(&self) -> Vec<u32> {
        self._shape()
    }

    /// Implements the addition of two `Tensor` instances. The `+` operator also does the same but the operator moves the value, making the instance unusable later.
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
    /// let tensor_sum = tensor_a.add(&tensor_b).expect("ShapeMismatch: Mismatch in shape of two Tensors.");
    /// // tensor_sum now contains the element-wise sum of tensor_a and tensor_b
    /// ```
    ///
    /// Note: This method supports all numeric types defined in the `numeric` module, allowing for a wide range of tensor operations.
    pub fn add(&self, rhs: &Self) -> Result<CpuTensor<T>, String> {
        self._add(rhs, false)
    }
    /// Computes the sum of all elements in the `Tensor` row-wise so that we get one column.
    pub fn sum(&self) -> CpuTensor<T> {
        let mut sum_vector = vec![T::zero(); self.shape[1] as usize];
        let rows = self.shape[0] as usize;
        let cols = self.shape[1] as usize;
        for r in 0..rows {
            for c in 0..cols {
                sum_vector[c] = sum_vector[c] + self.data[r * cols + c];
            }
        }
        CpuTensor::new(vec![1, self.shape[1]], sum_vector).unwrap()
    }

    /// Implements the subtraction of two `Tensor` instances. The `-` operator also does the same but the operator moves the value, making the instance unusable later.
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
    /// let tensor_diff = tensor_a.sub(&tensor_b).expect("ShapeMismatch: Mismatch in shape of two Tensors.");
    /// // tensor_diff now contains the element-wise difference of tensor_a and tensor_b
    /// ```
    ///
    /// Note: This method supports all numeric types defined in the `numeric` module, allowing for a wide range of tensor operations.
    pub fn sub(&self, rhs: &Self) -> Result<CpuTensor<T>, String> {
        self._add(rhs, true)
    }

    /// Implements tensor multiplication. The `*` also does the same but consumes the instance rendering it useless.
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
    /// let tensor_product = tensor_a.mul(&tensor_b).expect("ShapeMismatch: The dimensions of two matrices are not matching.");
    /// // tensor_product now contains the product of tensor_a and tensor_b
    /// ```
    ///
    /// Note: This method is designed to support all numeric types defined in the `numeric` module, ensuring compatibility with a wide range of data types.
    pub fn mul(&self, rhs: &Self) -> Result<CpuTensor<T>, String> {
        self._mul(rhs)
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
    /// - `Err(String)`: An error message if the shapes of the two tensors do not match.
    ///
    /// # Errors
    /// - `ShapeMismatch`: Returned when the shapes of the two tensors are not identical.
    ///
    /// # Example
    ///
    /// ```rust
    /// use iron_learn::CpuTensor;
    ///
    /// let tensor_a = CpuTensor::new(vec![2, 2], vec![1, 2, 3, 4]).unwrap();
    /// let tensor_b = CpuTensor::new(vec![2, 2], vec![5, 6, 7, 8]).unwrap();
    /// let tensor_product = tensor_a.multiply(&tensor_b).expect("ShapeMismatch: Mismatch in shape of two Tensors.");
    /// // tensor_product now contains the element-wise product of tensor_a and tensor_b
    /// ```
    ///
    /// Note: This method supports all numeric types defined in the `numeric` module, ensuring compatibility with a wide range of data types.
    pub fn multiply(&self, rhs: &Self) -> Result<Self, String> {
        self.hadamard(rhs)
    }

    /// Transposes a 2D tensor.
    ///
    /// This method reorganizes the data of a 2D tensor such that rows become columns
    /// and vice versa. For tensors with more than two dimensions, it returns an error.
    /// For 1D tensors, it simply clones the tensor as the transpose is the same.
    ///
    /// # Returns
    /// * `Ok(Self)` containing the transposed tensor if the tensor is 1D or 2D.
    /// * `Err(String)` containing an error message if the tensor has more than 2 dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// use iron_learn::CpuTensor;
    ///
    /// let tensor = CpuTensor::new(vec![2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();
    /// let transposed = tensor.t().unwrap();
    /// assert_eq!(CpuTensor::new(vec![3, 2], vec![1, 4, 2, 5, 3, 6]).unwrap(), transposed);
    /// ```
    ///
    /// # Errors
    /// If the tensor has more than 2 dimensions, an error is returned:
    ///
    pub fn t(&self) -> Result<Self, String> {
        self._t()
    }

    /// Scales the tensor by a scalar value.
    ///
    /// This method takes a scalar value of type `T` and scales each element of the tensor by this value.
    /// The result is a new tensor with the same shape as the original but with each element multiplied
    /// by the scalar.
    ///
    /// # Parameters
    /// * `scalar`: The value of type `T` to scale the tensor by.
    ///
    /// # Returns
    /// A new `Tensor` instance with scaled values.
    ///
    /// # Examples
    /// ```
    /// use iron_learn::CpuTensor;
    ///
    /// let tensor = CpuTensor::new(vec![1, 3], vec![1, 2, 3]).unwrap();
    /// let scaled_tensor = tensor.scale(2);
    /// assert_eq!(scaled_tensor.get_data(), vec![2, 4, 6]);
    /// ```
    pub fn scale(&self, scalar: T) -> Self {
        self._s(scalar)
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
        CpuTensor::new(self.shape, result).unwrap()
    }
}

#[cfg(test)]
#[test]
fn test_new() {
    let t = CpuTensor::new(vec![1u32, 2u32], vec![1i8, 2i8]).unwrap();

    assert_eq!(t.shape, vec![1u32, 2u32]);
    assert_eq!(t.data, vec![1i8, 2i8]);
}
