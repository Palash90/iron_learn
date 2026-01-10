use super::layers::*;
use crate::nn::loss_functions::LossFunctionType;
use crate::nn::LayerType;
use crate::nn::ModelData;
use crate::numeric::FloatingPoint;
use crate::tensor::math::TensorMath;
use crate::NeuralNet;
use crate::Tensor;
use colored::Colorize;

/// Builder type for constructing `NeuralNet` instances via a fluent API.
pub struct NeuralNetBuilder<T, D>
where
    T: TensorMath<D, MathOutput = T>,
    D: FloatingPoint,
{
    layers: Vec<Box<dyn Layer<T, D>>>,
}

impl<T, D> Default for NeuralNetBuilder<T, D>
where
    T: Tensor<D> + TensorMath<D, MathOutput = T> + 'static,
    D: FloatingPoint + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T, D> NeuralNetBuilder<T, D>
where
    T: Tensor<D> + TensorMath<D, MathOutput = T> + 'static,
    D: FloatingPoint + 'static,
{
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }
    /// Add a new fully-connected linear layer to the builder.
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

    /// Add an activation layer to the builder.
    pub fn add_activation(&mut self, act: LayerType, name: &str) {
        let layer = ActivationLayer::new(name, act);
        self.layers.push(Box::new(layer));
    }

    pub fn build_from_config(
        model: ModelData<D>,
        distribution: &DistributionType,
    ) -> NeuralNet<T, D> {
        let layers = model.layers;
        let loss = model.loss_fn_type;
        let name = model.name;

        let reconstructed_layers: Vec<Box<dyn Layer<T, D>>> = layers
            .iter()
            .map(|layer_data| match layer_data.layer_type {
                LayerType::Linear => Box::new(
                    LinearLayer::new(
                        layer_data.shape[0],
                        layer_data.shape[1],
                        &layer_data.name,
                        distribution,
                    )
                    .unwrap(),
                ) as Box<dyn Layer<T, D>>,
                _ => Box::new(ActivationLayer::new(
                    layer_data.name.as_str(),
                    layer_data.layer_type.clone(),
                )) as Box<dyn Layer<T, D>>,
            })
            .collect();

        Self::build_network(loss, &name, reconstructed_layers)
    }

    /// Finalize the builder and construct a `NeuralNet`.
    pub fn build(self, loss_fn_type: LossFunctionType, name: &String) -> NeuralNet<T, D> {
        Self::build_network(loss_fn_type, name, self.layers)
    }

    fn build_network(
        loss_fn_type: LossFunctionType,
        name: &String,
        layers: Vec<Box<dyn Layer<T, D>>>,
    ) -> NeuralNet<T, D> {
        println!("Building Network:");
        let mut parameter_count = 0;

        let layer_strings: Vec<String> = layers
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
            layers,
            loss_fn_type,
            parameter_count as u64,
            label,
            name.to_string(),
            0,
            D::zero(),
        )
    }

    pub fn build_from_model(model: ModelData<D>) -> NeuralNet<T, D> {
        println!("Restoring Model: {}", model.name.bold().green());
        let mut layers: Vec<Box<dyn Layer<T, D>>> = Vec::new();

        for layer_data in model.layers {
            match layer_data.layer_type {
                LayerType::Linear => {
                    let weight_tensor = T::new(layer_data.shape, layer_data.weights).unwrap();

                    println!(
                        "Building layer {} with weights {:?}",
                        layer_data.name,
                        weight_tensor.get_shape()
                    );

                    let layer = LinearLayer::from_data(weight_tensor, &layer_data.name);

                    layers.push(Box::new(layer));
                }
                _ => {
                    println!(
                        "Building layer {} of type {:?}",
                        layer_data.name, layer_data.layer_type
                    );

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
            model.loss_fn_type,
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
