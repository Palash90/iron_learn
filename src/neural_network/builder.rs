use crate::neural_network::core::ActivationLayer;
use crate::neural_network::core::Layer;
use crate::neural_network::core::LinearLayer;
use crate::tensor::math::TensorMath;
use crate::ActivationType;
use crate::LossFunction;
use crate::Tensor;
use crate::{NeuralNet, SignedNumeric};

pub struct NeuralNetBuilder<D, T>
where
    T: Tensor<D> + TensorMath<D, MathOutput = T>,
    D: SignedNumeric,
{
    layers: Vec<Box<dyn Layer<D, T>>>,
}

impl<D, T> NeuralNetBuilder<D, T>
where
    D: SignedNumeric + 'static,
    T: Tensor<D> + TensorMath<D, MathOutput = T> + 'static,
{
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn add_linear(&mut self, input_size: u32, output_size: u32, name: &str) {
        match LinearLayer::new(input_size, output_size, name) {
            Ok(layer) => self.layers.push(Box::new(layer)),
            Err(e) => {
                eprintln!("Error adding LinearLayer: {}", e);
            }
        }
    }

    pub fn add_activation(&mut self, act_type: ActivationType, name: &str) {
        let layer = ActivationLayer::new(act_type, name);
        self.layers.push(Box::new(layer));
    }

    pub fn build(self, loss_fn: Box<dyn LossFunction<D, T>>) -> NeuralNet<D, T> {
        NeuralNet {
            layers: self.layers,
            loss_fn,
        }
    }
}
