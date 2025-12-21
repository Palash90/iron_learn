use crate::tensor::math::TensorMath;
use crate::tensor::Tensor;
use rand::Rng;
use std::f32::consts::PI;


use crate::neural_network::ActivationFn;
use super::CoreNeuralNetType;
use crate::numeric::Numeric;
use crate::neural_network::LossFunction;

pub trait Layer<T>
where
    T: Tensor<CoreNeuralNetType> + TensorMath<CoreNeuralNetType, MathOutput = T> + 'static,
{
    fn forward(&mut self, input: &T) -> Result<T, String>;
    fn backward(&mut self, output_error: &T, learning_rate: CoreNeuralNetType)
        -> Result<T, String>;
    fn get_parameters(&self) -> Option<Vec<CoreNeuralNetType>> {
        None
    }
    fn name(&self) -> &str;
}

pub struct LinearLayer<T>
where
    T: Tensor<CoreNeuralNetType> + TensorMath<CoreNeuralNetType, MathOutput = T> + 'static,
{
    weights: T,
    input_cache: Option<T>,
    name: String,
}

impl<T> LinearLayer<T>
where
    T: Tensor<CoreNeuralNetType> + TensorMath<CoreNeuralNetType, MathOutput = T> + 'static,
{
    pub fn new(input_size: u32, output_size: u32, name: &str) -> Result<Self, String> {
        let mut rng = rand::rng();
        
        // Match Python's np.random.randn (Mean 0, StdDev 1)
        // Or stick to Xavier but ensure the math is identical
        let limit = (6.0 / (input_size + output_size) as f32).sqrt();
        
        let w_data: Vec<CoreNeuralNetType> = (0..(input_size * output_size))
            .map(|_| ((rng.random::<f32>() * 2.0 - 1.0) * limit).into())
            .collect();
        let b_data: Vec<CoreNeuralNetType> = vec![0.0.into(); output_size as usize];

        Ok(Self {
            weights: T::new(vec![input_size, output_size], w_data)?,
            input_cache: None,
            name: name.to_string(),
        })
    }
}

impl<T> Layer<T> for LinearLayer<T>
where
    T: Tensor<CoreNeuralNetType> + TensorMath<CoreNeuralNetType, MathOutput = T> + 'static,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn forward(&mut self, input: &T) -> Result<T, String> {
        self.input_cache = Some(input.add(&T::zeroes(input.get_shape()))?);
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

        weights_grad.print_matrix();

        // Update Parameters
        let w_step = weights_grad.scale(-lr)?;
        self.weights = self.weights.sub(&w_step)?;

        Ok(input_error)
    }

    fn get_parameters(&self) -> Option<Vec<CoreNeuralNetType>> {
        Some(self.weights.get_data())
    }
}



pub struct ActivationLayer<T>
where
    T: Tensor<CoreNeuralNetType> + TensorMath<CoreNeuralNetType, MathOutput = T> + 'static,
{
    activation: ActivationFn<T>,
    activation_prime: ActivationFn<T>,
    output_cache: Option<T>,
    name: String,
}

impl<T> ActivationLayer<T>
where
    T: Tensor<CoreNeuralNetType> + TensorMath<CoreNeuralNetType, MathOutput = T> + 'static,
{
    /// Now takes two function pointers as input
    pub fn new(
        activation: ActivationFn<T>,
        activation_prime: ActivationFn<T>,
        name: &str,
    ) -> Self {
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
    T: Tensor<CoreNeuralNetType> + TensorMath<CoreNeuralNetType, MathOutput = T> + 'static,
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

    fn backward(&mut self, output_error: &T, _lr: CoreNeuralNetType) -> Result<T, String> {
        let out = self.output_cache.as_ref()
            .ok_or_else(|| "No output cache found for backward pass".to_string())?;

        // Call the passed-in activation prime function
        // Note: Many derivatives (like sigmoid/tanh) use the output 'y' rather than input 'x'
        let prime = (self.activation_prime)(out)?;

        prime.multiply(output_error)
    }
}




pub struct NeuralNet<T>
where
    T: Tensor<CoreNeuralNetType> + TensorMath<CoreNeuralNetType, MathOutput = T> + 'static,
{
    pub layers: Vec<Box<dyn Layer<T>>>,
    pub loss_fn: Box<dyn LossFunction<CoreNeuralNetType, T>>,
}

impl<T> NeuralNet<T>
where
    T: Tensor<CoreNeuralNetType> + TensorMath<CoreNeuralNetType, MathOutput = T> + 'static,
{
    pub fn add(&mut self, layer: Box<dyn Layer<T>>) {
        self.layers.push(layer);
    }

    pub fn predict(&mut self, input: &T) -> Result<T, String> {
        let mut output = input.add(&T::zeroes(input.get_shape())).unwrap();

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
        F: FnMut(usize, CoreNeuralNetType, &mut Self),
    {
        let lr_min = 1e-6;

        for i in 0..epochs {
            println!("Epoch {}", i);

            let cos_term = ((PI as CoreNeuralNetType * i as CoreNeuralNetType)
                / ((epochs + epoch_offset) as CoreNeuralNetType))
                .cos();
            println!("cos {}", cos_term);

            let decay_factor = 0.5 * (1.0 + cos_term);
            let current_lr = lr_min + (base_lr.f32() - lr_min) * decay_factor;

            println!("Current lr {}", current_lr);

            let mut output = x_train.add(&T::zeroes(x_train.get_shape()))?;
            println!("Output");
            for layer in &mut self.layers {
                output = layer.forward(&output)?;
            }
            println!("Layer forwarded");

            let err = self.loss_fn.loss(y_train, &output);

            println!("Error calculated");

            let mut error_prime = self.loss_fn.loss_prime(y_train, &output)?;
            println!("Error Prime Calculated");

            for layer in self.layers.iter_mut().rev() {
                println!("\t\tLayer {} backward started", layer.name());
                error_prime = layer.backward(&error_prime, current_lr)?;
                println!("\t\tLayer {} backward completed", layer.name());
            }
            println!("Backward Error Primed");

            let hook_interval = match epochs > 1 {
                true => 1,
                false => epochs - 1,
            };


            // Hook (Periodic Reporting)
            if i == 0 || i % hook_interval == 0 {
                x_train.synchronize();
                let e_1 = err.unwrap();
                println!("E_1 calc");

                let e_2 = e_1.sum().unwrap();
                println!("E_2 calc");

                let err_val = e_2.get_data()[0];
                hook(i, err_val, self);
            }

            println!();
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

        println!("To be saved in {}", filepath)
    }
}
