// ============================================================================
// Internal Module Declarations
// ============================================================================

mod commons;
mod complex;
mod cpu_tensor;
mod cuda_tensor;
pub mod examples;
mod gpu_context;
mod gradient_descent;
pub mod neural_network;
mod numeric;
pub mod tensor;

// ============================================================================
// Types
// ============================================================================

pub use crate::complex::Complex;
pub use crate::numeric::{Numeric, SignedNumeric};
pub use crate::tensor::Tensor;

// ============================================================================
// Tensor Backend Re-exports
// ============================================================================

pub use crate::cpu_tensor::CpuTensor;

#[cfg(feature = "cuda")]
pub use crate::cuda_tensor::GpuTensor;

// ============================================================================
// Context & GPU Re-exports
// ============================================================================

#[cfg(feature = "cuda")]
pub use crate::gpu_context::{init_gpu, GpuContext, GPU_CONTEXT};

// ============================================================================
// Neural Network Re-exports
// ============================================================================

pub use crate::neural_network::{
    ActivationFn, ActivationLayer, Layer, LinearLayer, LossFunction, MeanSquaredErrorLoss,
    NeuralNet, NeuralNetBuilder,
};

// ============================================================================
// Data Processing Re-exports
// ============================================================================

pub use crate::commons::normalize_features;

// ============================================================================
// Gradient Descent Re-exports
// ============================================================================

pub use crate::gradient_descent::{
    gradient_descent, linear_regression, logistic_regression, predict_linear, predict_logistic,
};
