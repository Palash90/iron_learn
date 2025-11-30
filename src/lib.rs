//! # Iron Learn - A Rust Machine Learning Library
//!
//! A pure Rust, type-safe machine learning library in pure Rust with GPU acceleration.
//! Provides tensor operations, gradient-based optimization, and support for complex-valued computations.
//!
//! ## Key Features
//!
//! - **GPU-Accelerated Training**: CUDA kernels for efficient large-scale optimization
//! - **Type Safety**: All operations return `Result` types; no unwrap-induced panics
//! - **Zero-Copy Borrowing**: Compute without unnecessary allocations
//! - **Generic Numerics**: Works with f64, f32, i32, i64, Complex, and custom types
//! - **Automatic Preprocessing**: Feature normalization and bias handling built-in
//!
//! ## Modules
//!
//! - `tensor`: Core multi-dimensional array operations
//! - `complex`: Complex number arithmetic
//! - `numeric`: Generic type system for computations
//! - `gradient_descent`: CPU optimization algorithms
//! - `gpu_regression`: CUDA-accelerated training
//! - `regression`: High-level training interface
//! - `app_context`: Global application state
//!
//! ## Quick Example
//!
//! ```rust
//! use iron_learn::{Tensor, linear_regression};
//!
//! let x = Tensor::new(vec![4, 3], vec![0.0; 12])?;
//! let y = Tensor::new(vec![4, 1], vec![1.0; 4])?;
//! let mut w = Tensor::new(vec![4, 1], vec![0.0; 4])?;
//!
//! for _ in 0..1000 {
//!     w = linear_regression(&x, &y, &w, 0.01);
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Version
//!
//! 0.5.0 - GPU-Accelerated Release

mod app_context;
mod complex;
mod gpu_regression;
mod gradient_descent;
mod numeric;
mod read_file;
mod regression;
mod tensor;

pub use crate::app_context::{init_context, AppContext, GLOBAL_CONTEXT};
pub use crate::complex::Complex;
pub use crate::gpu_regression::{run_linear_cuda, run_logistics_cuda};
pub use crate::gradient_descent::gradient_descent;
pub use crate::gradient_descent::linear_regression;
pub use crate::gradient_descent::logistic_regression;
pub use crate::gradient_descent::{predict_linear, predict_logistic};
pub use crate::numeric::Numeric;
pub use crate::regression::Data;
pub use crate::regression::XY;
pub use crate::regression::{run_linear, run_logistic};
pub use crate::tensor::Tensor;
