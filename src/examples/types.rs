use crate::neural_network::NeuralNetDataType;
use serde::{Deserialize, Serialize};

/// Dataset container for single-precision (f32) examples.
///
/// Holds training and test matrices as flattened vectors along with
/// their dimensions. This is the primary structure used by the
/// CLI runners and model loaders for f32-based datasets.
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

/// Dataset container for double-precision (f64) examples.
///
/// Same shape and semantics as `Data`, but stores features and labels
/// with `f64` precision for algorithms or tests that require higher
/// numeric fidelity.
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
