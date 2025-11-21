//! # Iron Learn - Machine Learning Library
//!
//! A high-performance, pure Rust machine learning library featuring GPU-accelerated optimization,
//! comprehensive tensor operations, and support for complex-valued computations.
//!
//! ## Version
//! 0.5.0 - GPU Optmization Release
//!
//! ## Features
//!
//! - **GPU Acceleration**: CUDA-powered gradient descent with efficient kernel implementations
//! - **Tensor Framework**: Multi-dimensional tensor abstractions with generic numeric types
//! - **Optimization Algorithms**: Linear and logistic regression with feature normalization
//! - **Complex Number Support**: Native complex-valued tensor operations
//! - **Type Safety**: Comprehensive error handling with Result-based API
//!
//! ## Architecture
//!
//! The library is organized into specialized modules that work together seamlessly:
//!
//! - `tensor`: Core multi-dimensional array abstraction with linear algebra operations
//! - `complex`: Complex number arithmetic and tensor operations
//! - `numeric`: Generic numeric type system enabling type-agnostic algorithms
//! - `gradient_descent`: CPU optimization with feature normalization and regression variants
//! - `gpu_regression`: GPU-accelerated logistic regression using CUDA
//! - `regression`: High-level training and evaluation framework
//! - `app_context`: Global application state and GPU capability management
//!
//! ## Quick Start
//!
//! ```rust
//! use iron_learn::{Tensor, gradient_descent};
//!
//! // Create tensors for linear regression
//! let x = Tensor::new(vec![3, 2], vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0])?;
//! let y = Tensor::new(vec![3, 1], vec![5.0, 7.0, 9.0])?;
//! let w = Tensor::new(vec![2, 1], vec![0.1, 0.2])?;
//!
//! // Perform gradient descent step
//! let updated_w = gradient_descent(&x, &y, &w, 0.01, false);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
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
