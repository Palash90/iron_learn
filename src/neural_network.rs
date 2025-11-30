use std::vec;
use rand::thread_rng;
use rand::Rng;

use crate::normalizer::{denormalize_features, normalize_features, normalize_features_mean_std};
use crate::Data;
use crate::{tensor::Tensor, Numeric};

pub trait Layer {
    fn forward(&mut self, input: Tensor<f64>) -> Tensor<f64>;
    fn backward(&mut self, error: &Tensor<f64>, learning_rate: f64) -> Tensor<f64>;
    fn get_type(&self) -> &str;
}

struct LinearLayer {
    weights: Tensor<f64>,
    bias: Tensor<f64>,
    input: Option<Tensor<f64>>,
}

impl LinearLayer {
    fn new(features: u32, output_size: u32) -> Self {
        let fan_in = features as f64;
        let fan_out = output_size as f64;
        let limit = (6.0 / (fan_in + fan_out)).sqrt(); // Xavier uniform
        let mut rng = thread_rng();
        let mut w = Vec::with_capacity((features * output_size) as usize);
        for _ in 0..(features * output_size) {
            let val: f64 = rng.gen_range(-limit..limit);
            w.push(val);
        }

        let weights = Tensor::new(vec![features, output_size], w).unwrap();
        let bias = Tensor::new(vec![1, output_size], vec![0.0; output_size as usize]).unwrap();
        LinearLayer {
            weights,
            bias,
            input: None,
        }
    }
}

impl Layer for LinearLayer {
    fn get_type(&self) -> &str {
        "LinearLayer"
    }
    fn forward(&mut self, input: Tensor<f64>) -> Tensor<f64> {
        self.input = Some(input);
        let weighted_input = (self.input.clone()).unwrap().mul(&self.weights).unwrap();
        // Get the batch size from the weighted_input's shape (the first dimension)
        let batch_size = weighted_input.get_shape()[0];
        let output_size = weighted_input.get_shape()[1];

        // Prepare the data for the broadcasted bias
        let mut broadcasted_bias_data = Vec::with_capacity((batch_size * output_size) as usize);

        // Repeat the bias data for every sample in the batch
        for _ in 0..batch_size {
            let mut bias_data = self.bias.get_data().clone(); // Bias is [1, output_size]
            broadcasted_bias_data.append(&mut bias_data);
        }

        // Create the new broadcasted bias Tensor [BatchSize, OutputSize]
        let broadcasted_bias =
            Tensor::new(vec![batch_size, output_size], broadcasted_bias_data).unwrap();

        // 2. Add the broadcasted bias to the weighted input: (X * W) + B_broadcasted
        weighted_input.add(&broadcasted_bias).unwrap()
    }

    fn backward(&mut self, error: &Tensor<f64>, learning_rate: f64) -> Tensor<f64> {
        // error shape: [batch_size, output_size]
        // self.weights shape: [input_size, output_size]
        // self.input shape: [batch_size, input_size]

        // Gradient w.r.t. input: error × weights^T → [batch_size, output_size] × [output_size, input_size]
        let input_error = error.mul(&self.weights.t().unwrap()).unwrap();

        // Gradient w.r.t. weights: input^T × error → [input_size, batch_size] × [batch_size, output_size]
        let weights_error = self.input.clone().unwrap().t().unwrap().mul(error).unwrap();

        // Gradient w.r.t. bias: sum error across batch dimension
        let biases_error = error.sum();

        self.weights = self
            .weights
            .sub(&weights_error.scale(learning_rate))
            .unwrap();
        self.bias = self.bias.sub(&biases_error.scale(learning_rate)).unwrap();

        input_error
    }
}

struct ActivationLayer {
    activation: fn(&Tensor<f64>) -> Tensor<f64>,
    activation_derivative: fn(&Tensor<f64>) -> Tensor<f64>,
    input: Option<Tensor<f64>>,
    output: Option<Tensor<f64>>,
}

impl ActivationLayer {
    fn new(
        activation: fn(&Tensor<f64>) -> Tensor<f64>,
        activation_derivative: fn(&Tensor<f64>) -> Tensor<f64>,
    ) -> Self {
        ActivationLayer {
            activation,
            activation_derivative,
            input: None,
            output: None,
        }
    }
}

impl Layer for ActivationLayer {
    fn get_type(&self) -> &str {
        "ActivationLayer"
    }
    fn forward(&mut self, input: Tensor<f64>) -> Tensor<f64> {
        self.input = Some(input);
        let output = (self.activation)(&self.input.clone().unwrap());
        self.output = Some(output.clone());

        output
    }

    fn backward(&mut self, error: &Tensor<f64>, learning_rate: f64) -> Tensor<f64> {
        let input_derivative = (self.activation_derivative)(&self.input.clone().unwrap());

        input_derivative.hadamard(&error).unwrap()
    }
}

fn relu(x: &Tensor<f64>) -> Tensor<f64> {
    let data: Vec<f64> = x
        .get_data()
        .iter()
        .map(|&v| if v > 0.0 { v } else { 0.0 })
        .collect();
    Tensor::new(x.get_shape().clone(), data).unwrap()
}

fn relu_derivative(x: &Tensor<f64>) -> Tensor<f64> {
    let data: Vec<f64> = x
        .get_data()
        .iter()
        .map(|&v| if v > 0.0 { 1.0 } else { 0.0 })
        .collect();
    Tensor::new(x.get_shape().clone(), data).unwrap()
}

fn mse(predicted: &Tensor<f64>, actual: &Tensor<f64>) -> f64 {
    let diff = predicted.sub(actual).unwrap();
    let squared_diff = diff.mul(&diff).unwrap();
    squared_diff.sum().get_data()[0] / (predicted.get_data().len() as f64)
}

fn mse_derivative(predicted: &Tensor<f64>, actual: &Tensor<f64>) -> Tensor<f64> {
    let diff = predicted.sub(actual).unwrap();
    diff.scale(2.0 / (predicted.get_data().len() as f64))
}

pub struct NeuralNetwork {
    layers: Vec<Box<dyn Layer>>,
    loss: fn(&Tensor<f64>, &Tensor<f64>) -> f64,
    loss_derivative: fn(&Tensor<f64>, &Tensor<f64>) -> Tensor<f64>,
}

impl NeuralNetwork {
    pub fn new(
        loss: fn(&Tensor<f64>, &Tensor<f64>) -> f64,
        loss_derivative: fn(&Tensor<f64>, &Tensor<f64>) -> Tensor<f64>,
    ) -> Self {
        NeuralNetwork {
            layers: Vec::new(),
            loss,
            loss_derivative,
        }
    }

    pub fn add_layer(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
    }

    pub fn forward(&mut self, input: Tensor<f64>) -> Tensor<f64> {
        let mut output = input;
        for layer in self.layers.iter_mut() {
            output = layer.forward(output);
        }
        output
    }

    pub fn backward(&mut self, predicted: &Tensor<f64>, actual: &Tensor<f64>, learning_rate: f64) {
        let mut error = (self.loss_derivative)(predicted, actual);
        for layer in self.layers.iter_mut().rev() {
            error = layer.backward(&error, learning_rate);
        }
    }

    pub fn train(
        &mut self,
        x_train: &Tensor<f64>,
        y_train: &Tensor<f64>,
        epochs: u32,
        learning_rate: f64,
    ) {
        for i in 0..epochs {
            let predicted = self.forward(x_train.clone());

            self.backward(&predicted, y_train, learning_rate);
        }
    }
}

fn build_neural_net(features: u32, output_size: u32) -> NeuralNetwork {
    let mut nn = NeuralNetwork::new(mse, mse_derivative);

    nn.add_layer(Box::new(LinearLayer::new(features, 6)));
    nn.add_layer(Box::new(ActivationLayer::new(relu, relu_derivative)));

    nn.add_layer(Box::new(LinearLayer::new(6, 6)));
    nn.add_layer(Box::new(ActivationLayer::new(relu, relu_derivative)));

    nn.add_layer(Box::new(LinearLayer::new(6, output_size)));

    nn
}

pub fn run_neural_network() {
    // Placeholder for loading data
    let Data { linear: xy, .. } = crate::read_file::deserialize_data("data.json").unwrap();

    let x_train = Tensor::new(vec![xy.m, xy.n], xy.x.clone()).unwrap();
    let y_train = Tensor::new(vec![xy.m, 1], xy.y.clone()).unwrap();

    let (x_train, x_mean, x_std) = normalize_features_mean_std(&x_train);

    let (y_train, y_mean, y_std) = normalize_features_mean_std(&y_train);

    let epochs = 5000;
    let learning_rate = 0.01;

    let input_size = x_train.get_shape()[1];
    let output_size = y_train.get_shape()[1];

    let mut nn = build_neural_net(input_size, output_size);

    nn.train(&x_train, &y_train, epochs, learning_rate);

    // Initialize test data (the linear_regression function will handle normalization and bias)
    let x_test = Tensor::new(vec![xy.m_test, xy.n], xy.x_test.clone()).unwrap();
    let y_test = Tensor::new(vec![xy.m_test, 1], xy.y_test.clone()).unwrap();

    let x_test = normalize_features(&x_test, &x_mean, &x_std);

    // Make predictions using the trained weights
    let predictions = nn.forward(x_test);

    // Denormalize predictions
    let predictions = denormalize_features(&predictions, &y_mean, &y_std);

    // Calculate Mean Squared Error
    let mut total_squared_error = 0.0;
    let total = xy.m_test as usize;

    for i in 0..total {
        let pred = predictions.get_data()[i];
        let actual = y_test.get_data()[i];
        let error = pred - actual;
        total_squared_error += error * error;
    }

    let mse = total_squared_error / (total as f64);
    println!("\nResults:");
    println!("Total test samples: {}", total);
    println!("Mean Squared Error: {:.4}", mse);
    println!("Root MSE: {:.4}", mse.sqrt() as f64);
}
