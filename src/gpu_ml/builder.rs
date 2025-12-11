use super::layers::{ActivationDerivFn, ActivationFn, ActivationLayer, LinearLayer};
use cust::error::CudaResult;
use cust::module::Module;
use cust::prelude::*;
use cust::stream::Stream;

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
        GpuNetworkBuilder { layers: Vec::new() }
    }

    /// Add a linear layer to the network
    pub fn add_linear(&mut self, name: &str, input_size: usize, output_size: usize) {
        let layer = LinearLayer::new(name, input_size, output_size).unwrap();
        self.layers.push(NetworkLayer::Linear(layer));
    }

    /// Add an activation layer to the network
    pub fn add_activation(
        &mut self,
        name: &str,
        activation_fn: ActivationFn,
        derivative_fn: ActivationDerivFn,
    ) {
        let layer = ActivationLayer::new(name, activation_fn, derivative_fn);
        self.layers.push(NetworkLayer::Activation(layer));
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
