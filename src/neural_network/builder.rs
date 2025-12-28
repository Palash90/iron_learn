use super::layers::*;
use crate::neural_network::LayerType;
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

    pub fn add_linear(
        &mut self,
        input_size: u32,
        output_size: u32,
        name: &str,
        distribution: &DistributionType,
    ) {
        match LinearLayer::new(input_size, output_size, name, distribution) {
            Ok(layer) => self.layers.push(Box::new(layer)),
            Err(e) => {
                eprintln!("Error adding LinearLayer: {}", e);
            }
        }
    }

    pub fn add_activation(&mut self, act: LayerType, name: &str) {
        let layer = ActivationLayer::new(name, act);
        self.layers.push(Box::new(layer));
    }

    pub fn build(
        self,
        loss_fn: Box<dyn LossFunction<NeuralNetDataType, T>>,
        name: &String,
    ) -> NeuralNet<T> {
        println!("Building Network:");
        let mut parameter_count = 0;

        let layer_strings: Vec<String> = self
            .layers
            .iter()
            .map(|layer| {
                let name = layer.name();

                match layer.as_ref().get_parameters() {
                    Some(v) => {
                        let shape = v.get_shape();
                        parameter_count += shape.iter().product::<u32>() as u64;
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
        NeuralNet::new(
            self.layers,
            loss_fn,
            parameter_count as u64,
            label,
            name.to_string(),
            0,
            0.0,
        )
    }

    pub fn build_from_model(
        model: ModelData,
        loss_fn: Box<dyn LossFunction<NeuralNetDataType, T>>,
    ) -> NeuralNet<T> {
        println!("Restoring Model: {}", model.name.bold().green());
        let mut layers: Vec<Box<dyn Layer<T>>> = Vec::new();

        for layer_data in model.layers {
            match layer_data.layer_type {
                LayerType::Linear => {
                    let weight_tensor = T::new(layer_data.shape, layer_data.weights).unwrap();

                    let layer = LinearLayer::from_data(weight_tensor, &layer_data.name);

                    layers.push(Box::new(layer));
                }
                _ => {
                    let layer = ActivationLayer::new(&layer_data.name, layer_data.layer_type);
                    layers.push(Box::new(layer));
                }
            }
        }

        // Calculate parameter label (M, k, etc.) similar to your build method
        let param_count = model.parameter_count as usize;
        let label = if param_count >= 1_000_000 {
            format!("{:.1}M", param_count as f64 / 1_000_000.0)
        } else {
            format!("{}k", param_count / 1_000)
        };

        // Construct the final NeuralNet with restored state
        let net = NeuralNet::new(
            layers,
            loss_fn,
            model.parameter_count,
            label,
            model.name,
            model.epoch,
            model.saved_lr,
        );

        println!("Model restored successfully at Epoch {}", model.epoch);
        net
    }
}
