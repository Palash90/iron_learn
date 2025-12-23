// ============================================================================
// Internal Module Declarations
// ============================================================================

mod contexts;
mod commons;
mod complex;
mod cpu_tensor;
mod cuda_tensor;
mod gpu_context;
mod gradient_descent;
mod numeric;
mod read_file;
mod runners;

pub mod neural_network;
pub mod tensor;

// ============================================================================
// Type Re-exports
// ============================================================================

pub use crate::complex::Complex;
pub use crate::numeric::{Numeric, SignedNumeric};
pub use crate::tensor::Tensor;

// ============================================================================
// Tensor Backend Re-exports
// ============================================================================

pub use crate::cpu_tensor::CpuTensor;
pub use crate::cuda_tensor::GpuTensor;

// ============================================================================
// Context & GPU Re-exports
// ============================================================================

pub use crate::contexts::{init_context, AppContext, GLOBAL_CONTEXT};
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
pub use crate::runners::{run_linear, run_logistic, run_neural_net, Data, XY};

// ============================================================================
// Gradient Descent Re-exports
// ============================================================================

pub use crate::gradient_descent::{
    gradient_descent, linear_regression, logistic_regression, predict_linear, predict_logistic,
};
