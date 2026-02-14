use crate::nn::{get_activations, LayerType};
use crate::numeric::FloatingPoint;
use crate::tensor::math::TensorMath;
use crate::tensor::Tensor;

use rand_distr::{Distribution, Normal, Uniform};

/// Trait representing a neural network layer.
///
/// Layers must implement forward and backward passes and optionally expose
/// their parameters for inspection or serialization.
pub trait Layer<T, D>
where
    T: Tensor<D> + TensorMath<D, MathOutput = T> + 'static,
    D: FloatingPoint,
{
    fn forward(&mut self, input: &T, is_training: bool) -> Result<T, String>;
    fn backward(
        &mut self,
        output_error: &T,
        learning_rate: D,
        weight_normalization: bool,
    ) -> Result<T, String>;
    fn get_parameters(&self) -> Option<T> {
        None
    }
    fn name(&self) -> &str;
    fn layer_type(&self) -> &LayerType;
}

/// Fully-connected linear layer holding weights and an optional input cache.
pub struct LinearLayer<T, D>
where
    T: Tensor<D> + TensorMath<D, MathOutput = T> + 'static,
    D: FloatingPoint,
{
    weights: T,
    input_cache: Option<T>,
    name: String,

    layer_type: LayerType,
    _marker: std::marker::PhantomData<D>,
}

#[derive(Debug, Clone)]
/// Weight initialization distribution selector.
pub enum DistributionType {
    Xavier,
    Normal,
    Uniform,
    He,
}

impl<T, D> LinearLayer<T, D>
where
    T: Tensor<D> + TensorMath<D, MathOutput = T> + 'static,
    D: FloatingPoint,
{
    fn _initialize_weights(
        input_size: u32,
        output_size: u32,
        distribution: &DistributionType,
    ) -> Vec<D> {
        let mut rng = rand::rng();
        let normal = Normal::new(0.0, 1.0).unwrap();

        let limit =
            (D::from_f64(2.0) / (D::from_u32(input_size) + D::from_u32(output_size))).sqrt();
        let xavier = Normal::new(0.0, limit.f64()).unwrap();

        let uniform_limit = (6.0 / (input_size as f64 + output_size as f64)).sqrt();
        let uniform = Uniform::new(-uniform_limit, uniform_limit).unwrap();

        let he_limit = (2.0 / (input_size as f64)).sqrt();
        let he = Normal::new(0.0, he_limit).unwrap();

        let w_data: Vec<D> = (0..(input_size * output_size))
            .map(|_| match distribution {
                DistributionType::Uniform => D::from_f64(uniform.sample(&mut rng)),
                DistributionType::Xavier => D::from_f64(xavier.sample(&mut rng)),
                DistributionType::Normal => D::from_f64(normal.sample(&mut rng)),
                DistributionType::He => D::from_f64(he.sample(&mut rng)),
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
        let w_data = Self::_initialize_weights(input_size, output_size, distribution);

        let weights = T::new(vec![input_size, output_size], w_data).unwrap();
        //let weights = add_bias_term(&weights).unwrap();

        Ok(Self {
            weights,
            input_cache: None,
            name: name.to_string(),
            layer_type: LayerType::Linear,
            _marker: std::marker::PhantomData,
        })
    }

    pub fn from_data(weights: T, name: &str) -> Self {
        Self {
            weights,
            input_cache: None,
            name: name.to_string(),
            layer_type: LayerType::Linear,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T, D> Layer<T, D> for LinearLayer<T, D>
where
    T: Tensor<D> + TensorMath<D, MathOutput = T> + 'static,
    D: FloatingPoint,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn forward(&mut self, input: &T, is_training: bool) -> Result<T, String> {
        if is_training {
            self.input_cache = Some(input.add(&T::zeroes(input.get_shape()))?);
        }
        let matmul = input.matmul(&self.weights)?;
        Ok(matmul)
    }

    fn backward(&mut self, output_error: &T, lr: D, normalize_grad: bool) -> Result<T, String> {
        let input = self.input_cache.as_ref().ok_or("No forward pass cache!")?;

        // Calculate Input Error: error * weights.T
        let w_t = self.weights.t()?;
        let input_error = output_error.matmul(&w_t)?;

        // Calculate Weights Gradient: input.T * error
        let input_t = input.t()?;

        let weights_grad = input_t.matmul(output_error)?;

        // Update Parameters
        let w_step = match normalize_grad {
            true => {
                let decay_term = self.weights.scale(D::from_f64(0.1))?;
                let regularized_grad = weights_grad.add(&decay_term)?;
                regularized_grad.scale(lr)?
            }
            false => weights_grad.scale(lr)?,
        };
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
pub struct ActivationLayer<T, D>
where
    T: Tensor<D> + TensorMath<D, MathOutput = T> + 'static,
    D: FloatingPoint,
{
    layer_type: LayerType,
    output_cache: Option<T>,
    name: String,
    _marker: std::marker::PhantomData<D>,
}

impl<T, D> ActivationLayer<T, D>
where
    T: Tensor<D> + TensorMath<D, MathOutput = T> + 'static,
    D: FloatingPoint,
{
    /// Now takes two function pointers as input
    pub fn new(name: &str, layer_type: LayerType) -> Self {
        Self {
            output_cache: None,
            name: name.to_string(),
            layer_type,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T, D> Layer<T, D> for ActivationLayer<T, D>
where
    T: Tensor<D> + TensorMath<D, MathOutput = T> + 'static,
    D: FloatingPoint,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn forward(&mut self, input: &T, is_training: bool) -> Result<T, String> {
        let (activation, _) = get_activations(&self.layer_type);
        let output = (activation)(input)?;

        if is_training {
            self.output_cache = Some(output.add(&T::zeroes(output.get_shape()))?);
        }
        Ok(output)
    }

    fn backward(&mut self, output_error: &T, _lr: D, _: bool) -> Result<T, String> {
        let out = self
            .output_cache
            .as_ref()
            .ok_or_else(|| "No output cache found for backward pass".to_string())?;

        let (_, activation_prime) = get_activations(&self.layer_type);

        // Call the passed-in activation prime function
        // Note: Many derivatives (like sigmoid/tanh) use the output 'y' rather than input 'x'
        let prime = (activation_prime)(out)?;

        prime.mul(output_error)
    }

    fn layer_type(&self) -> &LayerType {
        &self.layer_type
    }
}
