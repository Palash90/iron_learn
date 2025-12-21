use super::layers::*;
use crate::neural_network::ActivationFn;
use crate::neural_network::CoreNeuralNetType;
use crate::neural_network::LossFunction;
use crate::tensor::math::TensorMath;
use crate::NeuralNet;
use crate::Tensor;

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

    pub fn add_activation(&mut self, act: ActivationFn<T>, act_prime: ActivationFn<T>, name: &str) {
        let layer = ActivationLayer::new(act, act_prime, name);
        self.layers.push(Box::new(layer));
    }

    pub fn build(self, loss_fn: Box<dyn LossFunction<CoreNeuralNetType, T>>) -> NeuralNet<T> {
        NeuralNet {
            layers: self.layers,
            loss_fn,
        }
    }
}
