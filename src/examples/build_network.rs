use crate::nn::DistributionType;
use crate::nn::LayerType;
use crate::numeric::FloatingPoint;
use crate::tensor::math::TensorMath;
use crate::NeuralNet;
use crate::NeuralNetBuilder;
use crate::Tensor;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use crate::examples::types::NetworkConfig;

/// Loads a network configuration from a JSON file and builds the NeuralNet.
pub fn build_neural_net_from_config<T, D>(
    name: &str,
    distribution: &DistributionType,
) -> NeuralNet<T, D>
where
    T: Tensor<D> + TensorMath<D, MathOutput = T> + 'static,
    D: FloatingPoint + 'static,
{
    let path = &("model_outputs/".to_owned() + name + "/network.json");

    println!("Path: {}", path);

    let mut file = File::open(Path::new(path)).unwrap();
    let mut contents = String::new();
    let _ = file.read_to_string(&mut contents);

    let config: NetworkConfig = serde_json::from_str(&contents).unwrap();

    let mut nn = NeuralNetBuilder::<T, D>::new();

    for (in_size, out_size, layer_type, label) in config.layers {
        match layer_type {
            LayerType::Linear => {
                nn.add_linear(in_size, out_size, &label, distribution);
            }
            _ => {
                let activation_label = format!("Act_{}", label);
                nn.add_activation(layer_type, &activation_label);
            }
        }
    }

    nn.build(config.loss_function, &name.to_string())
}
