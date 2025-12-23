use super::NeuralNetDataType;
use crate::neural_network::{get_activations, ActivationFn, LayerData, LayerType};
use crate::tensor::math::TensorMath;
use crate::tensor::Tensor;

use rand_distr::{Distribution, StandardNormal};

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

pub struct LinearLayer<T>
where
    T: Tensor<NeuralNetDataType> + TensorMath<NeuralNetDataType, MathOutput = T> + 'static,
{
    weights: T,
    input_cache: Option<T>,
    name: String,
    layer_type: LayerType,
}

impl<T> LinearLayer<T>
where
    T: Tensor<NeuralNetDataType> + TensorMath<NeuralNetDataType, MathOutput = T> + 'static,
{
    fn _initialize_weights(input_size: u32, output_size: u32) -> Vec<NeuralNetDataType> {
        let mut rng = rand::rng(); //StdRng::seed_from_u64(42); // rand::rng();

        let limit = (6.0 / (input_size as NeuralNetDataType + output_size as NeuralNetDataType)).sqrt();

        let w_data: Vec<NeuralNetDataType> = (0..(input_size * output_size))
            //.map(|_| (rng.random::<NeuralNetDataType>() * 2.0 - 1.0) * limit) // For Xavier
            .map(|_| {
                let val: NeuralNetDataType = StandardNormal.sample(&mut rng);
                val as NeuralNetDataType
            })
            .collect();

        w_data
    }
    pub fn new(
        input_size: u32,
        output_size: u32,
        name: &str,
        model_data: Option<LayerData>,
    ) -> Result<Self, String> {
        let w_data = match model_data {
            None => Self::_initialize_weights(input_size, output_size),
            Some(model_data) => {
                let incoming_input_size = model_data.shape[0];
                let incoming_output_size = model_data.shape[1];
                let incoming_name = model_data.name;

                if incoming_input_size == input_size
                    && incoming_output_size == output_size
                    && incoming_name.eq(name)
                {
                    model_data.weights
                } else {
                    Self::_initialize_weights(input_size, output_size)
                }
            }
        };

        Ok(Self {
            weights: T::new(vec![input_size, output_size], w_data)?,
            input_cache: None,
            name: name.to_string(),
            layer_type: LayerType::Linear,
        })
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
