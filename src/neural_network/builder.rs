use super::layers::*;
use crate::neural_network::ActivationFn;
use crate::neural_network::LossFunction;
use crate::neural_network::ModelData;
use crate::neural_network::NeuralNetDataType;
use crate::tensor::math::TensorMath;
use crate::NeuralNet;
use crate::Tensor;
use colored::Colorize;

pub struct NeuralNetBuilder<T>
where
    T: Tensor<NeuralNetDataType> + TensorMath<NeuralNetDataType, MathOutput = T>,
{
    layers: Vec<Box<dyn Layer<T>>>,
}

impl<T> NeuralNetBuilder<T>
where
    T: Tensor<NeuralNetDataType> + TensorMath<NeuralNetDataType, MathOutput = T> + 'static,
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

    pub fn build(
        self,
        loss_fn: Box<dyn LossFunction<NeuralNetDataType, T>>,
        network_model: Option<ModelData>,
    ) -> NeuralNet<T> {
        println!("Building Network:");
        let mut parameter_count = match network_model {
            None => 0,
            Some(v) => v.parameter_count,
        };

        let layer_strings: Vec<String> = self
            .layers
            .iter()
            .map(|layer| {
                let name = layer.name(); // Get the name (e.g., "Linear", "ReLU")

                match layer.as_ref().get_parameters() {
                    Some(v) => {
                        let shape = v.get_shape();
                        parameter_count += shape.iter().product::<u32>() as u64;
                        // Format: Name [In, Out]
                        format!("{} [{}, {}]", name.bold().cyan(), shape[0], shape[1])
                    }
                    None => {
                        // Format for Activation layers: just the Name
                        format!("{}", name.yellow())
                    }
                }
            })
            .collect();

        let parameter_count = parameter_count as usize;
        let label = if parameter_count >= 1_000_000 {
            format!("{:.1}M", parameter_count as f64 / 1_000_000.0)
        } else if parameter_count >= 1_000 {
            format!("{}k", parameter_count / 1_000)
        } else {
            format!("{}", parameter_count)
        };

        let output = layer_strings.join(" ──▶ ");

        println!("\nModel Architecture ({label}):");
        println!("{}\n", output);

        println!();
        NeuralNet {
            layers: self.layers,
            loss_fn,
            parameter_count: parameter_count as u64,
            label,
        }
    }
}
