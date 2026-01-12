// ============================================================================
// Public Module Declarations
// ============================================================================

pub mod builder;
pub mod loss_functions;

// ============================================================================
// Core Types
// ============================================================================

use crate::nn::loss_functions::LossFunctionType;
pub use builder::NeuralNetBuilder;
pub use neural_net::NeuralNet;
use serde::Deserialize;
use serde::Serialize;
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct LayerData<D>
where
    D: FloatingPoint,
{
    pub layer_type: LayerType,
    pub name: String,
    pub index: usize,
    pub weights: Vec<D>,
    pub shape: Vec<u32>,
}
/// Metadata describing a single layer when serializing/restoring models.
///
/// Stores the layer type, name, index within the network, raw weight
/// vector and its shape so layers can be reconstructed when loading a
/// saved model.
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ModelData<D>
where
    D: FloatingPoint,
{
    pub name: String,
    pub parameter_count: u64,
    pub layers: Vec<LayerData<D>>,
    pub epoch: usize,
    pub saved_lr: D,
    pub loss_fn_type: LossFunctionType,
    pub epoch_error: Vec<(usize, D, D)>,
    pub label: String,
}

// ============================================================================
// Type Definitions
// ============================================================================

/// Function pointer type for activation functions and their derivatives.
pub type ActivationFn<T> = fn(&T) -> Result<T, String>;

/// Function pointer type for loss functions and their derivatives.
pub type LossFn<T> = fn(&T, &T) -> Result<T, String>;

// ============================================================================
// Activation Functions
// ============================================================================

mod activations;
pub use activations::*;

// ============================================================================
// Layer Types and Traits
// ============================================================================

mod layers;
pub use layers::{ActivationLayer, DistributionType, Layer, LinearLayer};

use crate::numeric::FloatingPoint;

// ============================================================================
// Core Neural Network
// ============================================================================

mod neural_net;

pub mod types;
