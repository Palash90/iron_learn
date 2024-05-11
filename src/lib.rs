//! # Iron Learn
//! Iron Learn is a set of utilities to deal with simple machine learning tasks.

pub mod complex;

pub mod matrix;
pub mod numeric;
pub mod tensor;
pub mod vector;

pub use crate::complex::Complex;
pub use crate::matrix::Matrix;
pub use crate::tensor::Tensor;
pub use crate::vector::Vector;
