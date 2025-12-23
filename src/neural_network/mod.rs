// ============================================================================
// Public Module Declarations
// ============================================================================

pub mod builder;
pub mod loss_functions;

// ============================================================================
// Core Types
// ============================================================================

pub use builder::NeuralNetBuilder;
pub use loss_functions::{LossFunction, MeanSquaredErrorLoss};
pub use neural_net::NeuralNet;
use serde::Deserialize;
use serde::Serialize;
#[derive(Serialize, Deserialize)]
pub struct LayerData {
    pub layer_type: LayerType,
    pub name: String,
    pub index: usize,
    pub weights: Vec<NeuralNetDataType>,
    pub shape: Vec<u32>,
}

#[derive(Serialize, Deserialize)]
pub struct ModelData {
    pub name: String,
    pub parameter_count: u64,
    pub layers: Vec<LayerData>,
    pub epoch: usize,
}

// ============================================================================
// Type Definitions
// ============================================================================

pub type NeuralNetDataType = f32;
pub type ActivationFn<T> = fn(&T) -> Result<T, String>;

// ============================================================================
// Activation Functions
// ============================================================================

mod activations;
pub use activations::*;

// ============================================================================
// Layer Types and Traits
// ============================================================================

mod layers;
pub use layers::{ActivationLayer, Layer, LinearLayer};

// ============================================================================
// Core Neural Network
// ============================================================================

mod neural_net;
