use crate::tensor::Tensor;
use crate::Numeric;
use rand::Rng;
use std::f64::consts::PI;

type MyNumeric = f64;
type MyTensor = dyn Tensor<MyNumeric>;

pub trait LossFunction<T: Tensor<MyNumeric>> {
    /// Calculates the loss value (used for reporting).
    fn loss(&self, actual: &T, predicted: &T) -> Result<T, String>;

    /// Calculates the derivative of the loss w.r.t the predicted output (used for backpropagation).
    fn loss_prime(&self, actual: &T, predicted: &T) -> Result<T, String>;
}

pub struct MeanSquaredErrorLoss;

impl<T: Tensor<MyNumeric> + 'static> LossFunction<T> for MeanSquaredErrorLoss {
    // Loss: 0.5 * sum((actual - predicted)^2)
    fn loss(&self, actual: &T, predicted: &T) -> Result<T, String> {
        let error_diff = actual.sub(predicted)?;
        let sq_err = error_diff.multiply(&error_diff)?; // Element-wise square

        let sum_err = sq_err.sum()?; // Sum over all elements
        sum_err.scale(0.5) // Scale by 0.5
    }

    // Loss Prime: (predicted - actual)
    // The standard derivative is (predicted - actual) * 2 / N.
    // We omit the constant 2/N as it's absorbed into the learning rate.
    fn loss_prime(&self, actual: &T, predicted: &T) -> Result<T, String> {
        // Returns (predicted - actual)
        predicted.sub(actual)
    }
}

pub trait Layer<T: Tensor<MyNumeric>> {
    fn forward(&mut self, input: &T) -> Result<T, String>;
    fn backward(&mut self, output_error: &T, learning_rate: MyNumeric) -> Result<T, String>;
    fn get_parameters(&self) -> Option<(Vec<MyNumeric>, Vec<MyNumeric>)> {
        None
    }
    fn name(&self) -> &str;
}

pub struct LinearLayer<T: Tensor<MyNumeric>> {
    weights: T,
    biases: T,
    input_cache: Option<T>,
    name: String,
}

impl<T: Tensor<MyNumeric>> LinearLayer<T> {
    pub fn new(input_size: u32, output_size: u32, name: &str) -> Result<Self, String> {
        let mut rng = rand::thread_rng();
        let w_count = (input_size * output_size) as usize;

        // Initialize weights using f64
        let w_data: Vec<MyNumeric> = (0..w_count).map(|_| rng.gen::<MyNumeric>() - 0.5).collect();
        let weights = T::new(vec![input_size, output_size], w_data)?;

        // Initialize biases using f64
        let b_data = vec![0.0; output_size as usize];
        let biases = T::new(vec![1, output_size], b_data)?;

        println!(
            "{} = W shape: {:?}, b shape: {:?}",
            name,
            weights.get_shape(),
            biases.get_shape()
        );

        Ok(Self {
            weights,
            biases,
            input_cache: None,
            name: name.to_string(),
        })
    }
}

impl<T: Tensor<MyNumeric>> Layer<T> for LinearLayer<T> {
    fn name(&self) -> &str {
        &self.name
    }

    fn forward(&mut self, input: &T) -> Result<T, String> {
        self.input_cache = Some(input.add(&T::empty(input.get_shape()))?);

        println!(
            "Input Shape: {:?}, Cache Shape: {:?}",
            input.get_shape(),
            self.input_cache.as_ref().unwrap().get_shape()
        );

        let matmul = input.mul(&self.weights)?;

        println!("{} Matmul done", self.name);
        let output = matmul.add(&self.biases)?;
        Ok(output)
    }

    fn backward(&mut self, output_error: &T, lr: MyNumeric) -> Result<T, String> {
        let input = self.input_cache.as_ref().ok_or("No forward pass cache!")?;

        // Calculate Input Error: error * weights.T
        let w_t = self.weights.t()?;
        let input_error = output_error.mul(&w_t)?;

        // Calculate Weights Gradient: input.T * error
        let input_t = input.t()?;
        let weights_grad = input_t.mul(output_error)?;

        // Calculate Biases Gradient: sum(error)
        let biases_grad = output_error.sum()?;

        // Update Parameters
        let w_step = weights_grad.scale(-lr)?;
        self.weights = self.weights.add(&w_step)?;

        let b_step = biases_grad.scale(-lr)?;
        self.biases = self.biases.add(&b_step)?;

        Ok(input_error)
    }

    fn get_parameters(&self) -> Option<(Vec<MyNumeric>, Vec<MyNumeric>)> {
        Some((self.weights.get_data(), self.biases.get_data()))
    }
}

#[derive(Debug)]
pub enum ActivationType {
    Sigmoid,
    Tanh,
}

pub struct ActivationLayer<T: Tensor<MyNumeric>> {
    act_type: ActivationType,
    output_cache: Option<T>,
    name: String,
}

impl<T: Tensor<MyNumeric>> ActivationLayer<T> {
    pub fn new(act_type: ActivationType, name: &str) -> Self {
        println!("{} = {:?}", name, act_type);

        Self {
            act_type,
            output_cache: None,
            name: name.to_string(),
        }
    }
}

impl<T: Tensor<MyNumeric>> Layer<T> for ActivationLayer<T> {
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

    fn backward(&mut self, output_error: &T, _lr: MyNumeric) -> Result<T, String> {
        let out = self.output_cache.as_ref().unwrap();

        let prime = match self.act_type {
            ActivationType::Sigmoid => out.sigmoid(),
            ActivationType::Tanh => out.tanh(),
        };

        prime?.multiply(output_error)
    }
}

pub struct NeuralNet<T: Tensor<MyNumeric>> {
    pub layers: Vec<Box<dyn Layer<T>>>,
    loss_fn: Box<dyn LossFunction<T>>,
}

impl<T: Tensor<MyNumeric>> NeuralNet<T> {
    pub fn add(&mut self, layer: Box<dyn Layer<T>>) {
        self.layers.push(layer);
    }

    pub fn predict(&mut self, input: &T) -> Result<T, String> {
        let mut output = input.add(&T::empty(input.get_shape()))?;

        for layer in &mut self.layers {
            output = layer.forward(&output)?;
        }
        Ok(output)
    }

    pub fn fit<F>(
        &mut self,
        x_train: &T,
        y_train: &T,
        epochs: usize,
        epoch_offset: usize,
        base_lr: MyNumeric,
        mut hook: F,
    ) -> Result<(), String>
    where
        F: FnMut(usize, MyNumeric),
    {
        let lr_min = 1e-6;

        println!("Epochs {}", epochs);

        for i in 0..epochs {
            println!("{}", i);
            let cos_term = (PI * (i as MyNumeric) / ((epochs + epoch_offset) as MyNumeric)).cos();
            let decay_factor = 0.5 * (1.0 + cos_term);
            let current_lr = lr_min + (base_lr - lr_min) * decay_factor;

            println!("{} {} {}", cos_term, decay_factor, current_lr);

            let output = match self.predict(x_train) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("{}", e.to_string());
                    panic!();
                }
            };

            println!("predicted");

            let mut grad = self.loss_fn.loss_prime(y_train, &output)?;

            println!("Before Back propagates");

            for layer in self.layers.iter_mut().rev() {
                grad = layer.backward(&grad, current_lr)?;
            }

            println!("Back propagates");

            // Hook (Periodic Reporting)
            if i == 0 || (i + 1) % 1000 == 0 {
                let error_diff = y_train.sub(&output)?;
                let sq_err = error_diff.multiply(&error_diff)?;
                let sum_err = sq_err.sum()?;
                let err_val = sum_err.get_data()[0];
                hook(i, err_val);

                x_train.synchronize();
            }
        }

        x_train.synchronize();
        Ok(())
    }

    pub fn save_weights(&self, filepath: &str) {
        println!("Saving weights to {}...", filepath);
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some((w, b)) = layer.get_parameters() {
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

pub struct NeuralNetBuilder<T: Tensor<MyNumeric>> {
    layers: Vec<Box<dyn Layer<T>>>,
}

impl<T: Tensor<MyNumeric> + 'static> NeuralNetBuilder<T> {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn add_linear(mut self, input_size: u32, output_size: u32, name: &str) -> Self {
        match LinearLayer::new(input_size, output_size, name) {
            Ok(layer) => self.layers.push(Box::new(layer)),
            Err(e) => {
                eprintln!("Error adding LinearLayer: {}", e);
            }
        }
        self
    }

    pub fn add_activation(mut self, act_type: ActivationType, name: &str) -> Self {
        let layer = ActivationLayer::new(act_type, name);
        self.layers.push(Box::new(layer));
        self
    }

    pub fn build(self, loss_fn: Box<dyn LossFunction<T>>) -> NeuralNet<T> {
        NeuralNet {
            layers: self.layers,
            loss_fn,
        }
    }
}
