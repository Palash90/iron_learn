use crate::neural_network::core::ActivationLayer;
use crate::neural_network::core::Layer;
use crate::neural_network::core::LinearLayer;
use crate::neural_network::CoreNeuralNetType;
use crate::tensor::math::TensorMath;
use crate::ActivationType;
use crate::LossFunction;
use crate::Tensor;
use crate::{NeuralNet, SignedNumeric};

pub struct NeuralNetBuilder<T>
where
    T: Tensor<CoreNeuralNetType> + TensorMath<CoreNeuralNetType, MathOutput = T>,
{
    layers: Vec<Box<dyn Layer<T>>>,
}

impl<T> NeuralNetBuilder<T>
where
    T: Tensor<CoreNeuralNetType> + TensorMath<CoreNeuralNetType, MathOutput = T> + 'static,
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

    pub fn build(self, loss_fn: Box<dyn LossFunction<CoreNeuralNetType, T>>) -> NeuralNet<T> {
        NeuralNet {
            layers: self.layers,
            loss_fn,
        }
    }
}
