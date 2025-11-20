//! The `tensor` module provides the foundational structures and operations
//! for linear algebra computations integral to machine learning applications.

mod display;
use crate::app_context::GLOBAL_CONTEXT;
use crate::numeric::{Numeric, SignedNumeric};
use crate::AppContext;
use cust::launch;
use cust::memory::{CopyDestination, DeviceBuffer};
use cust::module::Module;
use cust::stream::{Stream, StreamFlags};
use std::ops::{Add, Mul, Neg, Sub};
use std::time::Instant;

/// The `Tensor` structure is the cornerstone of this library, providing a comprehensive suite of mathematical operations
/// for the manipulation of multidimensional data. It is designed to be compatible with all numeric types defined in the `numeric` module,
/// ensuring versatility and broad applicability in various machine learning contexts.

#[derive(Debug, PartialEq)]
pub struct Tensor<T: Numeric> {
    shape: Vec<u32>,
    data: Vec<T>,
}
impl<T> Tensor<T>
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

// This is the actual implementation of all the operations. This is here to avoid the documentation comment clutter.
impl<T: Numeric> Tensor<T> {
    pub fn _exp(operand: &Self) -> Tensor<f64> {
        let result = operand.data.iter().map(|t| f64::exp(t.f64())).collect();
        Tensor::new(operand.shape.clone(), result).unwrap()
    }

    fn _t(&self) -> Result<Self, String> {
        if self.shape.len() > 2 {
            return Err("Only 2D tensors can be transposed.".to_string());
        }

        if self.shape.len() == 1 {
            return Ok(Self {
                shape: self.shape.clone(),
                data: self.data.clone(),
            });
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

    fn check_shape(shape: &[u32]) -> Option<Result<Tensor<T>, String>> {
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

        if size != data.len().try_into().unwrap() {
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

    fn _gpu_mul(&self, rhs: &Self) -> Result<Self, String> {
        let rows = self.shape[0] as usize;
        let cols = rhs.shape[1] as usize;
        let common_dim = self.shape[1] as usize;

        let mut data = vec![T::zero(); rows * cols];

        let mut h_a = Vec::<f64>::new();
        let mut h_b = Vec::<f64>::new();

        for i in 0..self.data.len() {
            h_a.push(self.data[i].f64());
        }

        for i in 0..rhs.data.len() {
            h_b.push(rhs.data[i].f64());
        }

        let d_a = match DeviceBuffer::from_slice(h_a.as_slice()) {
            Ok(buf) => buf,
            Err(e) => return Err(format!("CUDA Device Buffer Creation Error for LHS: {}", e)),
        };

        let d_b = match DeviceBuffer::from_slice(h_b.as_slice()) {
            Ok(buf) => buf,
            Err(e) => return Err(format!("CUDA Device Buffer Creation Error for RHS: {}", e)),
        };

        let d_c = match DeviceBuffer::from_slice(data.as_slice()) {
            Ok(buf) => buf,
            Err(e) => {
                return Err(format!(
                    "CUDA Device Buffer Creation Error for RESULT: {}",
                    e
                ))
            }
        };

        // PTX produced from kernels/matrix_mul.cu
        let ptx = include_str!("../kernels/matrix_mul.ptx");
        let module = Module::from_ptx(ptx, &[]).unwrap();
        let function = match module.get_function("matrixMulKernel") {
            Ok(func) => func,
            Err(e) => return Err(format!("CUDA Kernel Function Retrieval Error: {}", e)),
        };

        let stream = Stream::new(StreamFlags::NON_BLOCKING, 1i32.into()).unwrap();

        // Kernel launch params (must match TILE used in the .cu)
        const TILE: u32 = 32;
        let block = (TILE, TILE, 1);
        let grid_x = ((cols as u32) + TILE - 1) / TILE;
        let grid_y = ((rows as u32) + TILE - 1) / TILE;
        let grid = (grid_x, grid_y, 1);

        let start = Instant::now();
        unsafe {
            let result = launch!(
                function<<<grid, block, 0, stream>>>(
                    d_a.as_device_ptr(),
                    d_b.as_device_ptr(),
                    d_c.as_device_ptr(),
                    rows as i32,
                    common_dim as i32,
                    cols as i32,
                )
            );
            match result {
                Ok(_) => {}
                Err(e) => return Err(format!("CUDA Kernel Launch Error: {}", e)),
            }
        }
        match stream.synchronize() {
            Ok(_) => {}
            Err(e) => return Err(format!("CUDA Stream Synchronization Error: {}", e)),
        }

        let duration = start.elapsed();
        println!("GPU Matrix Multiplication Time: {:.2?}\n", duration);

        let result = d_c.copy_to(&mut data);

        match result {
            Ok(_) => {}
            Err(e) => return Err(format!("CUDA Device to Host Copy Error: {}", e)),
        }

        Ok(Tensor {
            shape: vec![rows as u32, cols as u32],
            data,
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

        Ok(Tensor {
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

        let context = match GLOBAL_CONTEXT.get() {
            Some(ctx) => ctx,
            None => {
                println!("Application Context Error: Global context is not initialized.",);
                &AppContext {
                    app_name: "default",
                    version: 1,
                    gpu_enabled: false,
                    context: None,
                }
            }
        };

        if context.gpu_enabled {
            return self._gpu_mul(rhs);
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
impl<T: Numeric> Tensor<T> {
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
    /// use iron_learn::Tensor;
    ///
    /// let tensor = Tensor::exp(&Tensor::new(vec![2, 2], vec![1, 2, 3, 4]).unwrap());
    /// ```
    ///
    pub fn exp(operand: &Self) -> Tensor<f64> {
        Self::_exp(operand)
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
    /// This method returns a copy of the data held within the `Tensor` as a vector of type `T`, where `T` encompasses all numeric types supported by the library.
    /// It is a fundamental method that allows for direct access to the tensor's data for further processing or analysis.
    ///
    /// # Returns
    /// A `Vec<T>` containing a copy of the tensor's data.
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

    /// Retrieves the shape of the `Tensor`.
    ///
    /// # Returns
    /// A `Vec<u32>` containing a copy of the tensor's shape.
    ///
    /// # Example
    ///
    /// ```rust
    /// use iron_learn::Tensor;
    ///
    /// let tensor = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
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
    /// use iron_learn::Tensor;
    ///
    /// let tensor_a = Tensor::new(vec![2, 2], vec![1, 2, 3, 4]).unwrap();
    /// let tensor_b = Tensor::new(vec![2, 2], vec![5, 6, 7, 8]).unwrap();
    /// let tensor_sum = tensor_a.add(&tensor_b).expect("ShapeMismatch: Mismatch in shape of two Tensors.");
    /// // tensor_sum now contains the element-wise sum of tensor_a and tensor_b
    /// ```
    ///
    /// Note: This method supports all numeric types defined in the `numeric` module, allowing for a wide range of tensor operations.
    pub fn add(&self, rhs: &Self) -> Result<Tensor<T>, String> {
        self._add(rhs, false)
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
    /// use iron_learn::Tensor;
    ///
    /// let tensor_a = Tensor::new(vec![2, 2], vec![1, 2, 3, 4]).unwrap();
    /// let tensor_b = Tensor::new(vec![2, 2], vec![5, 6, 7, 8]).unwrap();
    /// let tensor_diff = tensor_a.sub(&tensor_b).expect("ShapeMismatch: Mismatch in shape of two Tensors.");
    /// // tensor_diff now contains the element-wise difference of tensor_a and tensor_b
    /// ```
    ///
    /// Note: This method supports all numeric types defined in the `numeric` module, allowing for a wide range of tensor operations.
    pub fn sub(&self, rhs: &Self) -> Result<Tensor<T>, String> {
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
    /// use iron_learn::Tensor;
    ///
    /// let tensor_a = Tensor::new(vec![2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();
    /// let tensor_b = Tensor::new(vec![3, 2], vec![7, 8, 9, 10, 11, 12]).unwrap();
    /// let tensor_product = tensor_a.mul(&tensor_b).expect("ShapeMismatch: The dimensions of two matrices are not matching.");
    /// // tensor_product now contains the product of tensor_a and tensor_b
    /// ```
    ///
    /// Note: This method is designed to support all numeric types defined in the `numeric` module, ensuring compatibility with a wide range of data types.
    pub fn mul(&self, rhs: &Self) -> Result<Tensor<T>, String> {
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
    /// use iron_learn::Tensor;
    ///
    /// let tensor = Tensor::new(vec![2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();
    /// let transposed = tensor.t().unwrap();
    /// assert_eq!(Tensor::new(vec![3, 2], vec![1, 4, 2, 5, 3, 6]).unwrap(), transposed);
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
    /// use iron_learn::Tensor;
    ///
    /// let tensor = Tensor::new(vec![1, 3], vec![1, 2, 3]).unwrap();
    /// let scaled_tensor = tensor.scale(2);
    /// assert_eq!(scaled_tensor.get_data(), vec![2, 4, 6]);
    /// ```
    pub fn scale(&self, scalar: T) -> Self {
        self._s(scalar)
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
    /// - `Err(String)`: An error message if the shapes of the two tensors do not match.
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
        self._add(&rhs, false)
    }
}

impl<T: Numeric> Sub for Tensor<T> {
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
    /// use iron_learn::Tensor;
    ///
    /// let tensor_a = Tensor::new(vec![2, 2], vec![1, 2, 3, 4]).unwrap();
    /// let tensor_b = Tensor::new(vec![2, 2], vec![5, 6, 7, 8]).unwrap();
    /// let tensor_sum = (tensor_a - tensor_b).expect("ShapeMismatch: Mismatch in shape of two Tensors.");
    /// // tensor_sum now contains the element-wise sum of tensor_a and tensor_b
    /// ```
    ///
    /// Note: This method supports all numeric types defined in the `numeric` module, allowing for a wide range of tensor operations.
    fn sub(self, rhs: Self) -> Result<Self, String> {
        self._add(&rhs, true)
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

impl<T: SignedNumeric> Neg for Tensor<T> {
    type Output = Self;
    fn neg(self) -> Self {
        let result: Vec<T> = self.data.iter().map(|t| -*t).collect();
        Tensor::new(self.shape, result).unwrap()
    }
}

#[cfg(test)]
#[test]
fn test_new() {
    let t = Tensor::new(vec![1u32, 2u32], vec![1i8, 2i8]).unwrap();

    assert_eq!(t.shape, vec![1u32, 2u32]);
    assert_eq!(t.data, vec![1i8, 2i8]);
}
