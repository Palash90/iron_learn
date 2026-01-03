use crate::numeric::FloatingPoint;
use serde::{Deserialize, Serialize};

/// Dataset container for single-precision (f32) examples.
///
/// Holds training and test matrices as flattened vectors along with
/// their dimensions. This is the primary structure used by the
/// CLI runners and model loaders for f32-based datasets.
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(bound = "")]
pub struct Data<D>
where
    D: FloatingPoint,
{
    pub m: u32,
    pub n: u32,
    pub x: Vec<D>,
    pub y: Vec<D>,
    pub m_test: u32,
    pub x_test: Vec<D>,
    pub y_test: Vec<D>,
}
