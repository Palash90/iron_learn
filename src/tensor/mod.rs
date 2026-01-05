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
//! GPU acceleration is available through the `cuda` feature for large-scale workloads.

use crate::Numeric;
pub mod math;
/// Core tensor trait exposing the minimal API required by the ML library.
///
/// Implementations (CPU/GPU) must provide construction, basic
/// arithmetic, reductions and device synchronization primitives used by
/// higher-level neural network components.
pub trait Tensor<T: Numeric>: Sized {
    fn print_matrix(&self) -> () {
        let data = self.get_data();

        let rows = self.get_shape()[0] as usize;
        let cols = match self.get_shape().len() {
            2 => self.get_shape()[1] as usize,
            _ => 1,
        };

        for r in 0..rows {
            for c in 0..cols {
                print!("{:.6?}\t", data[r * cols + c]);
            }
            println!();
        }
    }

    /* Creation */
    /// Returns a Tensor with all zero values
    fn zeroes(shape: &Vec<u32>) -> Self;

    /// Returns a Tensor with all one values
    fn ones(shape: &Vec<u32>) -> Self;

    /// Creates a new tensor with the provided shape and data
    fn new(shape: Vec<u32>, data: Vec<T>) -> Result<Self, String>;

    /* Retrieval */
    /// Returns the shape of the Tensor
    fn get_shape(&self) -> &Vec<u32>;

    /// Returns the data
    fn get_data(&self) -> Vec<T>;

    /* Device API */
    /// Synchronize with GPU, if running on GPU
    fn synchronize();

    /* Matrix related operations */
    /// Adds two Tensors and returns a tensor
    fn add(&self, rhs: &Self) -> Result<Self, String>;

    /// Subtracts rhs from current tensor
    fn sub(&self, rhs: &Self) -> Result<Self, String>;

    /// Element wise multiplication
    fn mul(&self, rhs: &Self) -> Result<Self, String>;

    /// Transpose a 2D matrix, only supported upto 2D
    fn t(&self) -> Result<Self, String>;

    /// Matrix multiplication
    fn matmul(&self, rhs: &Self) -> Result<Self, String>;

    /// Division
    fn div(&self, rhs: &Self) -> Result<Self, String>;

    /// Element wise scaling
    fn scale(&self, scalar: T) -> Result<Self, String>;

    /// Clip values to min, max
    fn clip(&self, min: T, max: T) -> Result<Self, String>;

    /* Reducers */
    /// Reduces rows column wise
    fn sum(&self) -> Result<Self, String> {
        let data = self.get_data();
        let total: T = data.iter().fold(T::zero(), |acc, &x| acc + x);

        Self::new(vec![1], vec![total])
    }
}
