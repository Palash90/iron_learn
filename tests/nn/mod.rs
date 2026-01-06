pub mod test_act;
pub mod test_activations;
pub mod test_builder;
pub mod test_loss;
pub mod test_nn;

#[cfg(feature = "cuda")]
pub mod test_activations_cuda;

#[cfg(feature = "cuda")]
pub mod test_loss_cuda;
