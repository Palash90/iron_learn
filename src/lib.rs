mod app_context;
mod commons;
mod complex;
mod cpu_tensor;
mod cuda_tensor;
mod gradient_descent;
mod neural_network;
mod numeric;
mod read_file;
mod runners;
pub mod tensor;

pub use crate::app_context::{
    init_context, init_gpu, AppContext, GpuContext, GLOBAL_CONTEXT, GPU_CONTEXT,
};
pub use crate::commons::normalize_features;
pub use crate::complex::Complex;
pub use crate::cpu_tensor::CpuTensor;
pub use crate::cuda_tensor::GpuTensor;
pub use crate::gradient_descent::gradient_descent;
pub use crate::gradient_descent::linear_regression;
pub use crate::gradient_descent::logistic_regression;
pub use crate::gradient_descent::{predict_linear, predict_logistic};
pub use crate::numeric::Numeric;
pub use crate::numeric::SignedNumeric;
pub use crate::runners::Data;
pub use crate::runners::XY;
pub use crate::runners::{run_linear, run_logistic, run_neural_net};
pub use neural_network::ActivationType;
pub use neural_network::LossFunction;
pub use neural_network::MeanSquaredErrorLoss;
pub use neural_network::NeuralNet;
pub use neural_network::NeuralNetBuilder;
pub use tensor::Tensor;
