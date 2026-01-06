use crate::numeric::{FloatingPoint, Numeric, SignedNumeric};
use crate::tensor::math::TensorMath;
use crate::Tensor;
use std::ops::{Add, Mul, Neg, Sub};

/// The `CpuTensor` structure is the backend implementaion of the `Tensor` trait. The implementation uses CPU for calcualations.
/// Although, Auto Vectorization has been used whereever possible for parallel execeution.
///
/// # Examples
///
/// ```rust
/// use crate::iron_learn::Tensor;
/// use iron_learn::CpuTensor;
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
impl<T> CpuTensor<T> where T: Numeric {}

enum OpType {
    Exp = 0,
    Sin = 2,
    Cos = 3,
    Tan = 4,
    Tanh = 5,
    Sigmoid = 6,
    Log = 7,
    Ln = 8,
    GreaterThanZeroMask = 9,
    ReLU = 10,
}

// This is the actual implementation of all the operations. This is here to avoid the documentation comment clutter.
impl<T: Numeric> CpuTensor<T> {
    fn _sigmoid(t: T) -> T
    where
        T: FloatingPoint,
    {
        // Numerically stable sigmoid implementation
        // Avoids overflow for large positive/negative values
        if t.f32() >= 0.0 {
            T::one() / (T::one() + (-t).exp())
        } else {
            t.exp() / (T::one() + t.exp())
        }
    }

    fn element_op(&self, op_type: OpType) -> CpuTensor<T>
    where
        T: FloatingPoint,
    {
        let result = match op_type {
            OpType::Exp => self.data.iter().map(|t| T::exp(t)).collect(),

            OpType::Cos => self.data.iter().map(|t| T::cos(t)).collect(),

            OpType::Sin => self.data.iter().map(|t| T::sin(t)).collect(),

            OpType::Tan => self.data.iter().map(|t| T::tan(t)).collect(),

            OpType::Tanh => self.data.iter().map(|t| T::tanh(t)).collect(),

            OpType::Sigmoid => self.data.iter().map(|t| Self::_sigmoid(*t)).collect(),

            OpType::Log => self.data.iter().map(|t| T::log10(t)).collect(),

            OpType::Ln => self.data.iter().map(|t| T::ln(t)).collect(),

            OpType::ReLU => self
                .data
                .iter()
                .map(|t| if t.f32() > 0.0 { *t } else { T::zero() })
                .collect(),

            OpType::GreaterThanZeroMask => self
                .data
                .iter()
                .map(|t| if t.f32() > 0.0 { T::one() } else { T::zero() })
                .collect(),
        };

        CpuTensor::new(self.shape.clone(), result).unwrap()
    }

    fn _t(&self) -> Result<Self, String> {
        if self.shape.len() > 2 {
            return Err("Only upto 2D tensors can be transposed.".to_string());
        }

        if self.shape.len() == 1 {
            return Ok(self.clone());
        }

        let rows = self.shape[0] as usize;
        let cols = self.shape[1] as usize;
        let mut new_data = vec![T::zero(); self.data.len()];

        for (r, row_slice) in self.data.chunks_exact(cols).enumerate() {
            for (c, &val) in row_slice.iter().enumerate() {
                // New index: column becomes row, row becomes column
                new_data[c * rows + r] = val;
            }
        }

        Ok(Self {
            shape: vec![cols as u32, rows as u32],
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
        if self.shape != rhs.shape {
            return Err(format!("ShapeMismatch:The dimensions of two matrices are not compatible for addition/subtraction- {:?} {:?}", self.shape, rhs.shape));
        }

        let a = &self.data;
        let b = &rhs.data;
        let mut result = vec![T::zero(); a.len()];

        if a.len() == b.len() && a.len() == result.len() {
            // Helping compiler for SIMD
            for i in 0..self.data.len() {
                if sub {
                    result[i] = a[i] - b[i];
                } else {
                    result[i] = a[i] + b[i];
                }
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

    fn _clip(&self, min: T, max: T) -> Result<Self, String> {
        let data = self.get_data();
        let shape = self.get_shape().clone();

        let result = data
            .iter()
            .map(|a| {
                if *a < min {
                    min
                } else if *a > max {
                    max
                } else {
                    *a
                }
            })
            .collect();

        Self::new(shape, result)
    }

    fn hadamard(&self, rhs: &Self) -> Result<Self, String> {
        if self.shape != rhs.shape {
            return Err(format!(
                "ShapeMismatch: Mismatch in shape of two Tensors {:?} {:?}",
                self.shape, rhs.shape
            ));
        }

        let result: Vec<T> = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a.mul(*b))
            .collect();

        Ok(Self {
            shape: self.shape.clone(),
            data: result,
        })
    }

    fn _div(&self, rhs: &Self) -> Result<Self, String> {
        if self.shape != rhs.shape {
            return Err(format!(
                "ShapeMismatch: Mismatch in shape of two Tensors {:?} {:?}",
                self.shape, rhs.shape
            ));
        }

        let result: Vec<T> = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a.div(*b))
            .collect();

        Ok(Self {
            shape: self.shape.clone(),
            data: result,
        })
    }

    fn _cpu_mul(&self, rhs: &Self) -> Result<Self, String> {
        let (rows, common_dim) = (self.shape[0] as usize, self.shape[1] as usize);

        let cols = rhs.shape[1] as usize;

        let mut data = vec![T::zero(); rows * cols];

        for i in 0..rows {
            // SIMD specific change
            let out_row_offset = i * cols;

            for k in 0..common_dim {
                let aik = self.data[i * common_dim + k]; // Constant for the inner loop
                let rhs_row_offset = k * cols;
                let rhs_slice = &rhs.data[rhs_row_offset..rhs_row_offset + cols];
                let out_slice = &mut data[out_row_offset..out_row_offset + cols];

                for j in 0..cols {
                    //SIMD Specific change. Help compiler understand
                    out_slice[j] = out_slice[j] + aik * rhs_slice[j];
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

    /// Implements the addition of two `CpuTensor` instances.
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
    /// use crate::iron_learn::Tensor;
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
    /// use crate::iron_learn::Tensor;
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
    /// use crate::iron_learn::Tensor;
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
impl<T: Numeric> Tensor<T> for CpuTensor<T> {
    /* Creation */
    /// Creates a new tensor with the provided shape and data.
    /// Wires to the internal validation logic.
    fn new(shape: Vec<u32>, data: Vec<T>) -> Result<Self, String> {
        Self::_new(shape, data)
    }

    fn zeroes(shape: &Vec<u32>) -> Self {
        let data = vec![T::zero(); shape.iter().product::<u32>() as usize];
        Self::_new(shape.clone(), data).unwrap()
    }

    fn ones(shape: &Vec<u32>) -> Self {
        let data = vec![T::one(); shape.iter().product::<u32>() as usize];
        Self::_new(shape.clone(), data).unwrap()
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
    fn synchronize() {
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
        self.hadamard(rhs)
    }

    /// Transpose a 2D matrix.
    /// Wires to the internal transpose logic.
    fn t(&self) -> Result<Self, String> {
        self._t()
    }

    /// Hadamard product (element-wise multiplication).
    /// Wires to the internal Hadamard product logic.
    fn matmul(&self, rhs: &Self) -> Result<Self, String> {
        self._mul(rhs)
    }

    /// Element-wise division
    fn div(&self, rhs: &Self) -> Result<Self, String> {
        self._div(rhs)
    }

    /// Element wise scaling (multiplication by a scalar).
    fn scale(&self, scalar: T) -> Result<Self, String> {
        Ok(self._s(scalar))
    }

    /// Clamp the value
    fn clip(&self, min: T, max: T) -> Result<Self, String> {
        self._clip(min, max)
    }
}

impl<T> TensorMath<T> for CpuTensor<T>
where
    T: FloatingPoint,
{
    type MathOutputElem = T;
    type MathOutput = CpuTensor<T>;

    fn sigmoid(&self) -> Result<Self::MathOutput, String> {
        Ok(self.element_op(OpType::Sigmoid))
    }

    fn log(&self) -> Result<Self::MathOutput, String> {
        Ok(self.element_op(OpType::Log))
    }

    fn ln(&self) -> Result<Self::MathOutput, String> {
        Ok(self.element_op(OpType::Ln))
    }

    fn sin(&self) -> Result<Self::MathOutput, String> {
        Ok(self.element_op(OpType::Sin))
    }

    fn cos(&self) -> Result<Self::MathOutput, String> {
        Ok(self.element_op(OpType::Cos))
    }

    fn tan(&self) -> Result<Self::MathOutput, String> {
        Ok(self.element_op(OpType::Tan))
    }

    fn tanh(&self) -> Result<Self::MathOutput, String> {
        Ok(self.element_op(OpType::Tanh))
    }

    fn exp(&self) -> Result<Self::MathOutput, String> {
        Ok(self.element_op(OpType::Exp))
    }

    fn relu(&self) -> Result<Self::MathOutput, String> {
        Ok(self.element_op(OpType::ReLU))
    }

    fn greater_than_zero_mask(&self) -> Result<Self::MathOutput, String> {
        Ok(self.element_op(OpType::GreaterThanZeroMask))
    }
}
