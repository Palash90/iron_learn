pub mod builder;
mod core;
pub mod loss_functions;

pub use core::ActivationType;
pub use core::NeuralNet;

pub use builder::NeuralNetBuilder;
pub use loss_functions::LossFunction;
pub use loss_functions::MeanSquaredErrorLoss;
