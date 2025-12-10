use super::layers::{LinearLayer, ActivationLayer, ActivationFn, ActivationDerivFn};
use cust::prelude::*;
use cust::module::Module;
use cust::stream::Stream;
use std::sync::Arc;
use cust::error::CudaResult;

/// Layer wrapper that can be either linear or activation
pub enum NetworkLayer {
    Linear(LinearLayer),
    Activation(ActivationLayer),
}

impl NetworkLayer {
    pub fn name(&self) -> &str {
        match self {
            NetworkLayer::Linear(l) => &l.name,
            NetworkLayer::Activation(a) => &a.name,
        }
    }
}

/// A builder for composing GPU neural networks layer by layer
pub struct GpuNetworkBuilder {
    layers: Vec<NetworkLayer>,
}

impl GpuNetworkBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        GpuNetworkBuilder {
            layers: Vec::new(),
        }
    }

    /// Add a linear layer to the network
    pub fn add_linear(
        mut self,
        name: &str,
        input_size: usize,
        output_size: usize,
    ) -> CudaResult<Self> {
        let layer = LinearLayer::new(name, input_size, output_size)?;
        self.layers.push(NetworkLayer::Linear(layer));
        Ok(self)
    }

    /// Add an activation layer to the network
    pub fn add_activation(
        mut self,
        name: &str,
        activation_fn: ActivationFn,
        derivative_fn: ActivationDerivFn,
    ) -> Self {
        let layer = ActivationLayer::new(name, activation_fn, derivative_fn);
        self.layers.push(NetworkLayer::Activation(layer));
        self
    }

    /// Get the current number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get layer by index
    pub fn get_layer(&self, index: usize) -> Option<&NetworkLayer> {
        self.layers.get(index)
    }

    /// Get mutable reference to layer by index
    pub fn get_layer_mut(&mut self, index: usize) -> Option<&mut NetworkLayer> {
        self.layers.get_mut(index)
    }

    /// Build the network and return it
    pub fn build(self) -> GpuNetwork {
        GpuNetwork {
            layers: self.layers,
        }
    }
}

impl Default for GpuNetworkBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// A fully constructed GPU neural network
pub struct GpuNetwork {
    pub layers: Vec<NetworkLayer>,
}

impl GpuNetwork {
    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get layer by index
    pub fn get_layer(&self, index: usize) -> Option<&NetworkLayer> {
        self.layers.get(index)
    }

    /// Get mutable reference to layer
    pub fn get_layer_mut(&mut self, index: usize) -> Option<&mut NetworkLayer> {
        self.layers.get_mut(index)
    }

    /// Print network architecture
    pub fn print_architecture(&self) {
        println!("\n=== GPU Neural Network Architecture ===");
        for (i, layer) in self.layers.iter().enumerate() {
            println!("Layer {}: {}", i, layer.name());
        }
        println!("=====================================\n");
    }

    /// Get all layer names
    pub fn layer_names(&self) -> Vec<&str> {
        self.layers.iter().map(|l| l.name()).collect()
    }

    /// Save network weights to host memory (only for linear layers)
    pub fn save_weights(&self) -> Vec<(String, Vec<f64>, Vec<f64>)> {
        self.layers
            .iter()
            .enumerate()
            .filter_map(|(i, layer)| {
                if let NetworkLayer::Linear(l) = layer {
                    if let Ok((w, b)) = l.save_parameters() {
                        return Some((format!("Layer_{}", i), w, b));
                    }
                }
                None
            })
            .collect()
    }

    /// Load network weights from host memory
    pub fn load_weights(&mut self, weights: Vec<(String, Vec<f64>, Vec<f64>)>) -> CudaResult<()> {
        for (_, w, b) in weights {
            for layer in &mut self.layers {
                if let NetworkLayer::Linear(l) = layer {
                    match l.load_parameters(w.clone(), b.clone()) {
                        Ok(_) => break,
                        Err(_) => continue,
                    }
                }
            }
        }
        Ok(())
    }
}

// Preset network configurations

/// Build a simple 2-layer network for regression
pub fn build_simple_2layer(
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
) -> CudaResult<GpuNetwork> {
    use super::activations::{relu_activation, relu_derivative};

    let network = GpuNetworkBuilder::new()
        .add_linear("Input_to_Hidden", input_size, hidden_size)?
        .add_activation("Hidden_ReLU", relu_activation, relu_derivative)
        .add_linear("Hidden_to_Output", hidden_size, output_size)?
        .build();
    
    Ok(network)
}

/// Build a 3-layer network with more capacity
pub fn build_3layer(
    input_size: usize,
    hidden1_size: usize,
    hidden2_size: usize,
    output_size: usize,
) -> CudaResult<GpuNetwork> {
    use super::activations::{relu_activation, relu_derivative};

    let network = GpuNetworkBuilder::new()
        .add_linear("Input_to_Hidden1", input_size, hidden1_size)?
        .add_activation("Hidden1_ReLU", relu_activation, relu_derivative)
        .add_linear("Hidden1_to_Hidden2", hidden1_size, hidden2_size)?
        .add_activation("Hidden2_ReLU", relu_activation, relu_derivative)
        .add_linear("Hidden2_to_Output", hidden2_size, output_size)?
        .build();
    
    Ok(network)
}

/// Build a deep 5-layer network (similar to Python's build_neural_net)
pub fn build_deep_network(
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
) -> CudaResult<GpuNetwork> {
    use super::activations::{relu_activation, relu_derivative};

    let network = GpuNetworkBuilder::new()
        .add_linear("Input_to_Hidden1", input_size, hidden_size)?
        .add_activation("Hidden1_ReLU", relu_activation, relu_derivative)
        .add_linear("Hidden1_to_Hidden2", hidden_size, hidden_size)?
        .add_activation("Hidden2_ReLU", relu_activation, relu_derivative)
        .add_linear("Hidden2_to_Hidden3", hidden_size, hidden_size * 2)?
        .add_activation("Hidden3_ReLU", relu_activation, relu_derivative)
        .add_linear("Hidden3_to_Hidden4", hidden_size * 2, hidden_size)?
        .add_activation("Hidden4_ReLU", relu_activation, relu_derivative)
        .add_linear("Hidden4_to_Hidden5", hidden_size, hidden_size / 2)?
        .add_activation("Hidden5_ReLU", relu_activation, relu_derivative)
        .add_linear("Hidden5_to_Output", hidden_size / 2, output_size)?
        .build();
    
    Ok(network)
}
