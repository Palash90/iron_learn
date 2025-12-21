pub mod builder;
mod core;
pub mod loss_functions;

pub use core::NeuralNet;

pub use builder::NeuralNetBuilder;
pub use loss_functions::LossFunction;
pub use loss_functions::MeanSquaredErrorLoss;

pub type CoreNeuralNetType = f32;
pub type ActivationFn<T> = fn(&T) -> Result<T, String>;

mod activations;
pub use activations::*;
