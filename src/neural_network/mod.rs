mod core;
pub mod loss_functions;

pub use core::ActivationType;
pub use core::NeuralNet;
pub use core::NeuralNetBuilder;

pub use loss_functions::LossFunction;
pub use loss_functions::MeanSquaredErrorLoss;