//! # Iron Learn
//! A pure Rust Machine Learning Library
//!
//! ## Status
//! Version 0.3.0 released with limited Matrix Manipulation abilities. Under active development for further implementation support.
//!
//! ## Overview
//! This library is designed to facilitate machine learning tasks with a focus on linear algebra operations.
//! Currently, the library supports matrix addition, subtraction,  multiplication,
//! transpose, scaling by a scalar and a gradient descent function providing a robust foundation for building more complex
//! machine learning algorithms.
//!

mod app_context;
mod complex;
mod gpu_regression;
mod gradient_descent;
mod numeric;
mod regression;
mod tensor;

pub use crate::app_context::{init_context, AppContext, GLOBAL_CONTEXT};
pub use crate::complex::Complex;
pub use crate::gpu_regression::run_ml_cuda;
pub use crate::gradient_descent::gradient_descent;
pub use crate::gradient_descent::linear_regression;
pub use crate::gradient_descent::logistic_regression;
pub use crate::gradient_descent::{predict_linear, predict_logistic};
pub use crate::numeric::Numeric;
pub use crate::regression::Data;
pub use crate::regression::XY;
pub use crate::regression::{run_linear, run_logistic};
pub use crate::tensor::Tensor;
