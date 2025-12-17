use crate::tensor::math::TensorMath;
use crate::tensor::Tensor;
use crate::SignedNumeric;
use rand::Rng;
use std::f32::consts::PI;

use crate::LossFunction;

use super::CoreNeuralNetType;
use crate::numeric::Numeric;

pub trait Layer<D, T>
where
    T: Tensor<D> + TensorMath<D, MathOutput = T> + 'static,
    D: SignedNumeric,
{
    fn forward(&mut self, input: &T) -> Result<T, String>;
    fn backward(&mut self, output_error: &T, learning_rate: CoreNeuralNetType) -> Result<T, String>;
    fn get_parameters(&self) -> Option<Vec<D>> {
        None
    }
    fn name(&self) -> &str;
}

pub struct LinearLayer<D, T>
where
    T: Tensor<D> + TensorMath<D, MathOutput = T> + 'static,
    D: SignedNumeric,
{
    weights: T,
    input_cache: Option<T>,
    name: String,
}

impl<D, T> LinearLayer<D, T>
where
    T: Tensor<D> + TensorMath<D, MathOutput = T> + 'static,
    D: SignedNumeric,
{
    pub fn new(input_size: u32, output_size: u32, name: &str) -> Result<Self, String> {
        let mut rng = rand::rng();
        let w_count = (input_size * output_size) as usize;

        let w_data: Vec<CoreNeuralNetType> = (0..w_count)
            .map(|_| rng.random::<CoreNeuralNetType>() - 0.5)
            .collect();
        let weights = T::new(vec![input_size, output_size], w_data)?;

        Ok(Self {
            weights,
            input_cache: None,
            name: name.to_string(),
        })
    }
}

impl<D, T> Layer<D, T> for LinearLayer<D, T>
where
    T: Tensor<D> + TensorMath<D, MathOutput = T> + 'static,
    D: SignedNumeric,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn forward(&mut self, input: &T) -> Result<T, String> {
        self.input_cache = Some(input.add(&T::empty(input.get_shape()))?);
        let matmul = input.mul(&self.weights)?;
        Ok(matmul)
    }

    fn backward(&mut self, output_error: &T, lr: CoreNeuralNetType) -> Result<T, String> {
        let input = self.input_cache.as_ref().ok_or("No forward pass cache!")?;

        // Calculate Input Error: error * weights.T
        let w_t = self.weights.t()?;
        let input_error = output_error.mul(&w_t)?;

        // Calculate Weights Gradient: input.T * error
        let input_t = input.t()?;
        let weights_grad = input_t.mul(output_error)?;

        // Update Parameters
        let w_step = weights_grad.scale(-lr)?;
        self.weights = self.weights.add(&w_step)?;

        Ok(input_error)
    }

    fn get_parameters(&self) -> Option<Vec<D>> {
        Some(self.weights.get_data())
    }
}

#[derive(Debug)]
pub enum ActivationType {
    Sigmoid,
    Tanh,
}

pub struct ActivationLayer<D, T>
where
    T: Tensor<D> + TensorMath<D, MathOutput = T> + 'static,
    D: SignedNumeric,
{
    act_type: ActivationType,
    output_cache: Option<T>,
    name: String,
}

impl<D, T> ActivationLayer<D, T>
where
    T: Tensor<D> + TensorMath<D, MathOutput = T> + 'static,
    D: SignedNumeric,
{
    pub fn new(act_type: ActivationType, name: &str) -> Self {
        Self {
            act_type,
            output_cache: None,
            name: name.to_string(),
        }
    }
}

impl<D, T> Layer<D, T> for ActivationLayer<D, T>
where
    D: SignedNumeric,
    T: Tensor<D> + TensorMath<D, MathOutput = T> + 'static,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn forward(&mut self, input: &T) -> Result<T, String> {
        let output = match self.act_type {
            ActivationType::Sigmoid => input.sigmoid()?,
            ActivationType::Tanh => input.tanh()?,
        };
        self.output_cache = Some(output.add(&T::empty(output.get_shape()))?);
        Ok(output)
    }

    fn backward(&mut self, output_error: &T, _lr: CoreNeuralNetType) -> Result<T, String> {
        let out = self.output_cache.as_ref().unwrap();

        let prime = match self.act_type {
            ActivationType::Sigmoid => {
                let sigmoid = out.sigmoid();
                sigmoid
            }
            ActivationType::Tanh => {
                let o_squared = out.multiply(out).unwrap();
                let ones = T::ones(&out.get_shape().clone());
                let tanh = ones.sub(&o_squared);
                tanh
            }
        };

        prime.unwrap().multiply(output_error)
    }
}

pub struct NeuralNet<D, T>
where
    T: Tensor<D> + TensorMath<D, MathOutput = T> + 'static,
    D: SignedNumeric,
{
    pub layers: Vec<Box<dyn Layer<D, T>>>,
    pub loss_fn: Box<dyn LossFunction<D, T>>,
}

impl<D, T> NeuralNet<D, T>
where
    T: Tensor<D> + TensorMath<D, MathOutput = T> + 'static,
    D: SignedNumeric,
{
    pub fn add(&mut self, layer: Box<dyn Layer<D, T>>) {
        self.layers.push(layer);
    }

    pub fn predict(&mut self, input: &T) -> Result<T, String> {
        let mut output = input.add(&T::empty(input.get_shape())).unwrap();

        for layer in &mut self.layers {
            output = layer.forward(&output).unwrap();
        }
        Ok(output)
    }

    pub fn fit<F>(
        &mut self,
        x_train: &T,
        y_train: &T,
        epochs: usize,
        epoch_offset: usize,
        base_lr: CoreNeuralNetType,
        mut hook: F,
    ) -> Result<(), String>
    where
        F: FnMut(usize, D, &mut Self),
    {
        let lr_min = 1e-6;

        for i in 0..epochs {
            let cos_term = ((PI * i as CoreNeuralNetType)
                / ((epochs + epoch_offset) as CoreNeuralNetType))
                .cos();
            let decay_factor = 0.5 * (1.0 + cos_term);
            let current_lr = lr_min + (base_lr.f32() - lr_min) * decay_factor;

            let mut output = x_train.add(&T::empty(x_train.get_shape()))?;
            for layer in &mut self.layers {
                output = layer.forward(&output)?;
            }

            let err = self.loss_fn.loss(y_train, &output);

            let mut error = self.loss_fn.loss_prime(y_train, &output)?;
            for layer in self.layers.iter_mut().rev() {
                error = layer.backward(&error, current_lr)?;
            }

            let hook_interval = match epochs > 1000 {
                true => 1000,
                false => epochs - 1,
            };

            // Hook (Periodic Reporting)
            if i == 0 || i % hook_interval == 0 {
                let error_diff = y_train.sub(&output)?;
                let sq_err = error_diff.multiply(&error_diff)?;
                let sum_err = sq_err.sum()?;
                let err_val = sum_err.get_data()[0];
                hook(i, err_val, self);

                x_train.synchronize();
            }
        }

        x_train.synchronize();
        Ok(())
    }

    pub fn save_weights(&self, filepath: &str) {
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(w) = layer.get_parameters() {
                println!(
                    "Layer {} ({}) Weights saved. (Shape: {:?})",
                    i,
                    layer.name(),
                    w.len()
                );
            }
        }
    }
}
