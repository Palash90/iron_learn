//! # Iron Learn
//! Iron Learn is a set of utilities to deal with simple machine learning tasks.

mod complex;
mod matrix;
mod numeric;
mod tensor;
mod vector;

pub use crate::complex::Complex;
pub use crate::matrix::Matrix;
pub use crate::numeric::Numeric;
pub use crate::tensor::Tensor;
pub use crate::vector::Vector;
