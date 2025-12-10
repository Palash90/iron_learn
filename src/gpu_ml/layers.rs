use cust::error::CudaResult;
use cust::memory::DeviceBuffer;
use cust::module::Module;
use cust::prelude::*;
use cust::stream::Stream;

/// Type alias for the activation function signature.
/// Takes (input_ptr, output_ptr, size, module, stream) and applies activation.
pub type ActivationFn = fn(
    d_input: DevicePointer<f64>,
    d_output: DevicePointer<f64>,
    size: i32,
    module: &Module,
    stream: &Stream,
) -> CudaResult<()>;

/// Type alias for activation derivative.
/// Takes (z_ptr, deriv_output_ptr, size, module, stream) and computes derivative.
pub type ActivationDerivFn = fn(
    d_z: DevicePointer<f64>,
    d_deriv: DevicePointer<f64>,
    size: i32,
    module: &Module,
    stream: &Stream,
) -> CudaResult<()>;

/// Metadata for a layer (name, type, dimensions)
#[derive(Clone, Debug)]
pub struct LayerInfo {
    pub name: String,
    pub layer_type: LayerType,
    pub input_size: usize,
    pub output_size: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub enum LayerType {
    Linear,
    ReLU,
    Sigmoid,
    Tanh,
    Identity,
}

/// Linear layer: computes output = input @ weights
pub struct LinearLayer {
    pub name: String,
    pub input_size: usize,
    pub output_size: usize,
    pub weights: DeviceBuffer<f64>,
    pub biases: DeviceBuffer<f64>,
}

impl LinearLayer {
    /// Create a new linear layer with xavier initialization
    pub fn new(name: &str, input_size: usize, output_size: usize) -> CudaResult<Self> {
        // Initialize weights with small random values (deterministic hash-based)
        let scale = (2.0 / input_size as f64).sqrt();
        let mut host_w = vec![0.0f64; input_size * output_size];
        let mut host_b = vec![0.0f64; output_size];

        for (i, w) in host_w.iter_mut().enumerate() {
            *w = ((i as f64).sin() * scale).min(0.1).max(-0.1);
        }
        for (i, b) in host_b.iter_mut().enumerate() {
            *b = ((i as f64).cos() * 0.01).min(0.001).max(-0.001);
        }

        let weights = DeviceBuffer::from_slice(&host_w)?;
        let biases = DeviceBuffer::from_slice(&host_b)?;

        Ok(LinearLayer {
            name: name.to_string(),
            input_size,
            output_size,
            weights,
            biases,
        })
    }

    /// Get mutable reference to weights for training
    pub fn weights_mut(&mut self) -> &mut DeviceBuffer<f64> {
        &mut self.weights
    }

    /// Get mutable reference to biases for training
    pub fn biases_mut(&mut self) -> &mut DeviceBuffer<f64> {
        &mut self.biases
    }

    /// Get immutable reference to weights
    pub fn weights(&self) -> &DeviceBuffer<f64> {
        &self.weights
    }

    /// Get immutable reference to biases
    pub fn biases(&self) -> &DeviceBuffer<f64> {
        &self.biases
    }

    /// Save weights and biases to host
    pub fn save_parameters(&self) -> CudaResult<(Vec<f64>, Vec<f64>)> {
        let mut w = vec![0.0f64; self.input_size * self.output_size];
        let mut b = vec![0.0f64; self.output_size];
        self.weights.copy_to(&mut w)?;
        self.biases.copy_to(&mut b)?;
        Ok((w, b))
    }

    /// Load weights and biases from host
    pub fn load_parameters(&mut self, weights: Vec<f64>, biases: Vec<f64>) -> CudaResult<()> {
        self.weights = DeviceBuffer::from_slice(&weights)?;
        self.biases = DeviceBuffer::from_slice(&biases)?;
        Ok(())
    }
}

/// Activation layer: applies activation function to input
/// This is more of a configuration/metadata holder than a data structure
pub struct ActivationLayer {
    pub name: String,
    pub activation_fn: ActivationFn,
    pub derivative_fn: ActivationDerivFn,
}

impl ActivationLayer {
    pub fn new(name: &str, activation_fn: ActivationFn, derivative_fn: ActivationDerivFn) -> Self {
        ActivationLayer {
            name: name.to_string(),
            activation_fn,
            derivative_fn,
        }
    }
}
