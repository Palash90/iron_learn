//! # Iron Learn
//! A pure Rust Machine Learning Library
//!
//! ## Status
//! Version 0.2.0 released with limited Matrix Manipulation abilities. Under active development for further implementation support.
//!
//! ## Overview
//! This library is designed to facilitate machine learning tasks with a focus on linear algebra operations. Currently, the library supports matrix addition, subtraction,  multiplication and transpose, providing a robust foundation for building more complex machine learning algorithms.
//!

mod complex;
pub mod gradient_descent;
mod matrix;
mod numeric;
mod tensor;
mod vector;

pub use crate::complex::Complex;
pub use crate::matrix::Matrix;
pub use crate::numeric::Numeric;
pub use crate::tensor::Tensor;
pub use crate::vector::Vector;
