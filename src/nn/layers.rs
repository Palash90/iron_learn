use super::NeuralNetDataType;
use crate::nn::{get_activations, LayerType};
use crate::tensor::math::TensorMath;
use crate::tensor::Tensor;

use rand_distr::{Distribution, Normal, Uniform};

/// Trait representing a neural network layer.
///
/// Layers must implement forward and backward passes and optionally expose
/// their parameters for inspection or serialization.
pub trait Layer<T>
where
    T: Tensor<NeuralNetDataType> + TensorMath<NeuralNetDataType, MathOutput = T> + 'static,
{
    fn forward(&mut self, input: &T) -> Result<T, String>;
    fn backward(&mut self, output_error: &T, learning_rate: NeuralNetDataType)
        -> Result<T, String>;
    fn get_parameters(&self) -> Option<T> {
        None
    }
    fn name(&self) -> &str;
    fn layer_type(&self) -> &LayerType;
}

/// Fully-connected linear layer holding weights and an optional input cache.
pub struct LinearLayer<T>
where
    T: Tensor<NeuralNetDataType> + TensorMath<NeuralNetDataType, MathOutput = T> + 'static,
{
    weights: T,
    input_cache: Option<T>,
    name: String,
    layer_type: LayerType,
}

#[derive(Debug, Clone)]
/// Weight initialization distribution selector.
pub enum DistributionType {
    Xavier,
    Normal,
    Uniform,
    He,
}

impl<T> LinearLayer<T>
where
    T: Tensor<NeuralNetDataType> + TensorMath<NeuralNetDataType, MathOutput = T> + 'static,
{
    fn _initialize_weights(
        input_size: u32,
        output_size: u32,
        distribution: &DistributionType,
    ) -> Vec<NeuralNetDataType> {
        let mut rng = rand::rng();
        let normal = Normal::new(0.0, 1.0).unwrap();

        let limit =
            (2.0 / (input_size as NeuralNetDataType + output_size as NeuralNetDataType)).sqrt();
        let xavier = Normal::new(0.0, limit).unwrap();

        let uniform_limit =
            (6.0 / (input_size as NeuralNetDataType + output_size as NeuralNetDataType)).sqrt();
        let uniform = Uniform::new(-uniform_limit, uniform_limit).unwrap();

        let he_limit = (2.0 / (input_size as NeuralNetDataType)).sqrt();
        let he = Normal::new(0.0, he_limit).unwrap();

        let w_data: Vec<NeuralNetDataType> = (0..(input_size * output_size))
            .map(|_| match distribution {
                DistributionType::Uniform => uniform.sample(&mut rng) as NeuralNetDataType,
                DistributionType::Xavier => xavier.sample(&mut rng) as NeuralNetDataType,
                DistributionType::Normal => normal.sample(&mut rng) as NeuralNetDataType,
                DistributionType::He => he.sample(&mut rng) as NeuralNetDataType,
            })
            .collect();

        w_data
    }
    pub fn new(
        input_size: u32,
        output_size: u32,
        name: &str,
        distribution: &DistributionType,
    ) -> Result<Self, String> {
        let w_data = Self::_initialize_weights(input_size, output_size, &distribution);

        let weights = T::new(vec![input_size, output_size], w_data).unwrap();
        //let weights = add_bias_term(&weights).unwrap();

        Ok(Self {
            weights,
            input_cache: None,
            name: name.to_string(),
            layer_type: LayerType::Linear,
        })
    }

    pub fn from_data(weights: T, name: &str) -> Self {
        Self {
            weights,
            input_cache: None,
            name: name.to_string(),
            layer_type: LayerType::Linear,
        }
    }
}

impl<T> Layer<T> for LinearLayer<T>
where
    T: Tensor<NeuralNetDataType> + TensorMath<NeuralNetDataType, MathOutput = T> + 'static,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn forward(&mut self, input: &T) -> Result<T, String> {
        self.input_cache = Some(input.add(&T::zeroes(input.get_shape()))?);
        let matmul = input.mul(&self.weights)?;
        Ok(matmul)
    }

    fn backward(&mut self, output_error: &T, lr: NeuralNetDataType) -> Result<T, String> {
        let input = self.input_cache.as_ref().ok_or("No forward pass cache!")?;

        // Calculate Input Error: error * weights.T
        let w_t = self.weights.t()?;
        let input_error = output_error.mul(&w_t)?;

        // Calculate Weights Gradient: input.T * error
        let input_t = input.t()?;

        let weights_grad = input_t.mul(output_error)?;

        // Update Parameters
        let w_step = weights_grad.scale(lr)?;
        self.weights = self.weights.sub(&w_step)?;

        Ok(input_error)
    }

    fn get_parameters(&self) -> Option<T> {
        Some(
            T::zeroes(self.weights.get_shape())
                .add(&self.weights)
                .unwrap(),
        )
    }

    fn layer_type(&self) -> &LayerType {
        &self.layer_type
    }
}

/// Activation wrapper layer that applies element-wise activation functions.
pub struct ActivationLayer<T>
where
    T: Tensor<NeuralNetDataType> + TensorMath<NeuralNetDataType, MathOutput = T> + 'static,
{
    layer_type: LayerType,
    output_cache: Option<T>,
    name: String,
}

impl<T> ActivationLayer<T>
where
    T: Tensor<NeuralNetDataType> + TensorMath<NeuralNetDataType, MathOutput = T> + 'static,
{
    /// Now takes two function pointers as input
    pub fn new(name: &str, layer_type: LayerType) -> Self {
        Self {
            output_cache: None,
            name: name.to_string(),
            layer_type,
        }
    }
}

impl<T> Layer<T> for ActivationLayer<T>
where
    T: Tensor<NeuralNetDataType> + TensorMath<NeuralNetDataType, MathOutput = T> + 'static,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn forward(&mut self, input: &T) -> Result<T, String> {
        let (activation, _) = get_activations(&self.layer_type);
        // Call the passed-in activation function
        let output = (activation)(input)?;

        // Caching the output for the backward pass
        self.output_cache = Some(output.add(&T::zeroes(output.get_shape()))?);
        Ok(output)
    }

    fn backward(&mut self, output_error: &T, _lr: NeuralNetDataType) -> Result<T, String> {
        let out = self
            .output_cache
            .as_ref()
            .ok_or_else(|| "No output cache found for backward pass".to_string())?;

        let (_, activation_prime) = get_activations(&self.layer_type);

        // Call the passed-in activation prime function
        // Note: Many derivatives (like sigmoid/tanh) use the output 'y' rather than input 'x'
        let prime = (activation_prime)(out)?;

        prime.multiply(output_error)
    }

    fn layer_type(&self) -> &LayerType {
        &self.layer_type
    }
}
