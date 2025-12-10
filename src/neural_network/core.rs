use crate::Numeric;
use std::f64::consts::PI;
use rand::Rng;
use crate::tensor::Tensor;


type MyNumeric = f64;
type MyTensor = dyn Tensor<MyNumeric>;

/// A common interface for all layers.
pub trait Layer<T: Tensor<MyNumeric>> {
    fn forward(&mut self, input: &T) -> Result<T, String>;
    fn backward(&mut self, output_error: &T, learning_rate: MyNumeric) -> Result<T, String>;
    fn get_parameters(&self) -> Option<(Vec<MyNumeric>, Vec<MyNumeric>)> { None }
    fn name(&self) -> &str;
}

// --- 1. The Linear Layer ---

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

        Ok(Self { weights, biases, input_cache: None, name: name.to_string() })
    }
}

impl<T: Tensor<MyNumeric>> Layer<T> for LinearLayer<T> {
    fn name(&self) -> &str { &self.name }

    fn forward(&mut self, input: &T) -> Result<T, String> {
        self.input_cache = Some(input.add(&T::empty())?); 
        let matmul = input.mul(&self.weights)?;
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

// --- 2. The Activation Layer ---

pub enum ActivationType { Sigmoid, Tanh }

pub struct ActivationLayer<T: Tensor<MyNumeric>> {
    act_type: ActivationType,
    output_cache: Option<T>, 
}

impl<T: Tensor<MyNumeric>> ActivationLayer<T> {
    pub fn new(act_type: ActivationType) -> Self {
        Self { act_type, output_cache: None }
    }
}

impl<T: Tensor<MyNumeric>> Layer<T> for ActivationLayer<T> {
    fn name(&self) -> &str { "Activation" }

    fn forward(&mut self, input: &T) -> Result<T, String> {
        let output = match self.act_type {
            ActivationType::Sigmoid => input.sigmoid()?,
            ActivationType::Tanh => input.tanh()?,
        };
        self.output_cache = Some(output.add(&T::empty())?); 
        Ok(output)
    }

    fn backward(&mut self, output_error: &T, _lr: MyNumeric) -> Result<T, String> {
        let out = self.output_cache.as_ref().unwrap();
        
        let prime = match self.act_type {
            ActivationType::Sigmoid => {
                // Sigmoid Prime: s * (1 - s)
                // Note: The '1.0' in the vector is now f64
                let shape = out.get_shape();
                let ones_data = vec![1.0; shape.iter().product::<u32>() as usize];
                let ones = T::new(shape.clone(), ones_data)?;
                
                let one_minus_s = ones.sub(out)?;
                out.multiply(&one_minus_s)? // Hadamard product
            },
            ActivationType::Tanh => T::empty() // Placeholder
        };

        // Element-wise multiplication: fPrime * error
        prime.multiply(output_error)
    }
}

// --- 3. The Neural Network Trainer ---

pub struct NeuralNet<T: Tensor<MyNumeric>> {
    pub layers: Vec<Box<dyn Layer<T>>>,
}

impl<T: Tensor<MyNumeric>> NeuralNet<T> {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn add(&mut self, layer: Box<dyn Layer<T>>) {
        self.layers.push(layer);
    }

    pub fn predict(&mut self, input: &T) -> Result<T, String> {
        let mut output = input.add(&T::empty())?;
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
        base_lr: MyNumeric, // base_lr is now f64
        mut hook: F, 
    ) -> Result<(), String> 
    where F: FnMut(usize, MyNumeric) // Hook receives f64 error
    {
        let lr_min = 1e-6;

        for i in 0..epochs {
            // LR Decay calculated using f64
            let cos_term = (PI * (i as MyNumeric) / ((epochs + epoch_offset) as MyNumeric)).cos();
            let decay_factor = 0.5 * (1.0 + cos_term);
            let current_lr = lr_min + (base_lr - lr_min) * decay_factor;

            // Forward and Backward Passes (Device)
            let output = self.predict(x_train)?;
            let mut grad = output.sub(y_train)?; 

            for layer in self.layers.iter_mut().rev() {
                grad = layer.backward(&grad, current_lr)?;
            }

            // Hook (Periodic Reporting)
            if i == 0 || (i + 1) % 100 == 0 {
                 let error_diff = y_train.sub(&output)?;
                 let sq_err = error_diff.multiply(&error_diff)?;
                 let sum_err = sq_err.sum()?; 
                 let err_val = sum_err.get_data()[0]; 
                 hook(i, err_val);
            }
        }
        
        x_train.synchronize(); 
        Ok(())
    }

    pub fn save_weights(&self, filepath: &str) {
        println!("Saving weights to {}...", filepath);
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some((w, b)) = layer.get_parameters() {
                println!("Layer {} ({}) Weights saved. (Shape: {:?})", i, layer.name(), w.len());
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

    pub fn add_activation(mut self, act_type: ActivationType) -> Self {
        let layer = ActivationLayer::new(act_type);
        self.layers.push(Box::new(layer));
        self
    }
    
    pub fn build(self) -> NeuralNet<T> {
        NeuralNet { layers: self.layers }
    }
}