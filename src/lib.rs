// ============================================================================
// Internal Module Declarations
// ============================================================================

mod commons;
mod complex;
mod contexts;
mod cpu_tensor;
mod cuda_tensor;
mod gpu_context;
mod gradient_descent;
mod numeric;
pub mod read_file;
mod runners;
use serde::{Deserialize, Serialize};
pub mod neural_network;
pub mod tensor;
use crate::neural_network::NeuralNetDataType;

// ============================================================================
// Types
// ============================================================================

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Data {
    pub m: u32,
    pub n: u32,
    pub x: Vec<NeuralNetDataType>,
    pub y: Vec<NeuralNetDataType>,
    pub m_test: u32,
    pub x_test: Vec<NeuralNetDataType>,
    pub y_test: Vec<NeuralNetDataType>,
}
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct DataDoublePrecision {
    pub m: u32,
    pub n: u32,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub m_test: u32,
    pub x_test: Vec<f64>,
    pub y_test: Vec<f64>,
}

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

pub use crate::contexts::{init_context, AppContext, GLOBAL_CONTEXT};

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
pub use crate::runners::{run_linear, run_logistic, run_neural_net};

// ============================================================================
// Gradient Descent Re-exports
// ============================================================================

pub use crate::gradient_descent::{
    gradient_descent, linear_regression, logistic_regression, predict_linear, predict_logistic,
};
