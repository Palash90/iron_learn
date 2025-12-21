use super::NeuralNetDataType;
use crate::neural_network::ActivationFn;
use crate::neural_network::LossFunction;
use crate::tensor::math::TensorMath;
use crate::tensor::Tensor;
use rand::Rng;

pub trait Layer<T>
where
    T: Tensor<NeuralNetDataType> + TensorMath<NeuralNetDataType, MathOutput = T> + 'static,
{
    fn forward(&mut self, input: &T) -> Result<T, String>;
    fn backward(&mut self, output_error: &T, learning_rate: NeuralNetDataType)
        -> Result<T, String>;
    fn get_parameters(&self) -> Option<Vec<NeuralNetDataType>> {
        None
    }
    fn name(&self) -> &str;
}

pub struct LinearLayer<T>
where
    T: Tensor<NeuralNetDataType> + TensorMath<NeuralNetDataType, MathOutput = T> + 'static,
{
    weights: T,
    input_cache: Option<T>,
    name: String,
}

impl<T> LinearLayer<T>
where
    T: Tensor<NeuralNetDataType> + TensorMath<NeuralNetDataType, MathOutput = T> + 'static,
{
    pub fn new(input_size: u32, output_size: u32, name: &str) -> Result<Self, String> {
        let mut rng = rand::rng();

        let w_data: Vec<NeuralNetDataType> = (0..(input_size * output_size))
            .map(|_| (rng.random::<NeuralNetDataType>() * 2.0 - 1.0).into())
            .collect();

        Ok(Self {
            weights: T::new(vec![input_size, output_size], w_data)?,
            input_cache: None,
            name: name.to_string(),
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
        let w_step = weights_grad.scale(-lr)?;
        self.weights = self.weights.sub(&w_step)?;

        Ok(input_error)
    }

    fn get_parameters(&self) -> Option<Vec<NeuralNetDataType>> {
        Some(self.weights.get_data())
    }
}

pub struct ActivationLayer<T>
where
    T: Tensor<NeuralNetDataType> + TensorMath<NeuralNetDataType, MathOutput = T> + 'static,
{
    activation: ActivationFn<T>,
    activation_prime: ActivationFn<T>,
    output_cache: Option<T>,
    name: String,
}

impl<T> ActivationLayer<T>
where
    T: Tensor<NeuralNetDataType> + TensorMath<NeuralNetDataType, MathOutput = T> + 'static,
{
    /// Now takes two function pointers as input
    pub fn new(activation: ActivationFn<T>, activation_prime: ActivationFn<T>, name: &str) -> Self {
        Self {
            activation,
            activation_prime,
            output_cache: None,
            name: name.to_string(),
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
        // Call the passed-in activation function
        let output = (self.activation)(input)?;

        // Caching the output for the backward pass
        self.output_cache = Some(output.add(&T::zeroes(output.get_shape()))?);
        Ok(output)
    }

    fn backward(&mut self, output_error: &T, _lr: NeuralNetDataType) -> Result<T, String> {
        let out = self
            .output_cache
            .as_ref()
            .ok_or_else(|| "No output cache found for backward pass".to_string())?;

        // Call the passed-in activation prime function
        // Note: Many derivatives (like sigmoid/tanh) use the output 'y' rather than input 'x'
        let prime = (self.activation_prime)(out)?;

        prime.multiply(output_error)
    }
}
