use rand::rng;
use rand::Rng;
use std::vec;

use crate::tensor::Tensor;
use crate::Data;

pub trait Layer<T: Tensor<f64>> {
    fn forward(&mut self, input: T) -> T;
    fn backward(&mut self, error: &T, learning_rate: f64) -> T;
    fn get_type(&self) -> &str;
}

struct LinearLayer<T: Tensor<f64>> {
    weights: T,
    bias: T,
    input: T,
}

impl<T: Tensor<f64>> LinearLayer<T> {
    fn new(features: u32, output_size: u32) -> Self {
        let fan_in = features as f64;
        let fan_out = output_size as f64;
        let limit = (6.0 / (fan_in + fan_out)).sqrt();
        let mut rng = rng();
        let mut w = Vec::with_capacity((features * output_size) as usize);
        for _ in 0..(features * output_size) {
            let val: f64 = rng.random_range(-limit..limit);
            w.push(val);
        }

        let weights = T::new(vec![features, output_size], w).unwrap();
        let bias = T::new(vec![1, output_size], vec![0.0; output_size as usize]).unwrap();
        LinearLayer {
            weights,
            bias,
            input: None,
        }
    }
}

impl<T: Tensor<f64>> Layer<T> for LinearLayer<T> {
    fn get_type(&self) -> &str {
        "LinearLayer"
    }
    fn forward(&mut self, input: T) -> T {
        self.input = Some(input);
        let weighted_input = (self.input.clone()).unwrap().mul(&self.weights).unwrap();

        let batch_size = weighted_input.get_shape()[0];
        let output_size = weighted_input.get_shape()[1];

        let mut broadcasted_bias_data = Vec::with_capacity((batch_size * output_size) as usize);

        for _ in 0..batch_size {
            let mut bias_data = self.bias.get_data().clone();
            broadcasted_bias_data.append(&mut bias_data);
        }

        let broadcasted_bias =
            T::new(vec![batch_size, output_size], broadcasted_bias_data).unwrap();

        weighted_input.add(&broadcasted_bias).unwrap()
    }

    fn backward(&mut self, error: &T, learning_rate: f64) -> T {
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
        self.bias = self
            .bias
            .sub(&(biases_error.unwrap().scale(learning_rate).unwrap()))
            .unwrap();

        input_error
    }
}

struct ActivationLayer<T: Tensor<f64>> {
    activation: fn(&T) -> T,
    activation_derivative: fn(&T) -> T,
    input: Option<T>,
    output: Option<T>,
}

impl<T: Tensor<f64>> ActivationLayer<T> {
    fn new(activation: fn(&T) -> T, activation_derivative: fn(&T) -> T) -> Self {
        ActivationLayer {
            activation,
            activation_derivative,
            input: None,
            output: None,
        }
    }
}

impl<T: Tensor<f64>> Layer<T> for ActivationLayer<T> {
    fn get_type(&self) -> &str {
        "ActivationLayer"
    }
    fn forward(&mut self, input: T) -> T {
        self.input = Some(input);
        let output = (self.activation)(&self.input.clone().unwrap());
        self.output = Some(output.clone());

        output
    }

    fn backward(&mut self, error: &T, learning_rate: f64) -> T {
        let input_derivative = (self.activation_derivative)(&self.input.clone().unwrap());

        input_derivative.multiply(&error).unwrap()
    }
}

fn log_loss<T: Tensor<f64>>(predicted: &T, actual: &T) -> f64 {
    let epsilon = 1e-15;
    let clipped_preds: Vec<f64> = predicted
        .get_data()
        .iter()
        .map(|&p| p.max(epsilon).min(1.0 - epsilon))
        .collect();
    let mut loss = 0.0;
    for (p, a) in clipped_preds.iter().zip(actual.get_data().iter()) {
        loss += -a * p.ln() - (1.0f64 - a) * (1.0f64 - p).ln();
    }
    loss / (predicted.get_data().len() as f64)
}
fn log_loss_derivative<T: Tensor<f64>>(predicted: &T, actual: &T) -> T {
    let epsilon = 1e-15;
    let clipped_preds: Vec<f64> = predicted
        .get_data()
        .iter()
        .map(|&p| p.max(epsilon).min(1.0 - epsilon))
        .collect();

    let data: Vec<f64> = clipped_preds
        .iter()
        .zip(actual.get_data().iter())
        .map(|(&p, &a)| -(a / p) + (1.0 - a) / (1.0 - p))
        .collect();

    T::new(predicted.get_shape().clone(), data).unwrap()
}
fn sigmoid<T: Tensor<f64>>(x: &T) -> T {
    let data: Vec<f64> = x
        .get_data()
        .iter()
        .map(|&v| 1.0 / (1.0 + (-v).exp()))
        .collect();
    T::new(x.get_shape().clone(), data).unwrap()
}

fn sigmoid_prime<T: Tensor<f64>>(x: &T) -> T {
    let s = sigmoid(x);
    let data: Vec<f64> = s.get_data().iter().map(|&v| v * (1.0 - v)).collect();
    T::new(x.get_shape().clone(), data).unwrap()
}

fn relu<T: Tensor<f64>>(x: &T) -> T {
    let data: Vec<f64> = x
        .get_data()
        .iter()
        .map(|&v| if v > 0.0 { v } else { 0.0 })
        .collect();
    T::new(x.get_shape().clone(), data).unwrap()
}

fn relu_derivative<T: Tensor<f64>>(x: &T) -> T {
    let data: Vec<f64> = x
        .get_data()
        .iter()
        .map(|&v| if v > 0.0 { 1.0 } else { 0.0 })
        .collect();
    T::new(x.get_shape().clone(), data).unwrap()
}

fn mse<T: Tensor<f64>>(predicted: &T, actual: &T) -> f64 {
    let diff = predicted.sub(actual).unwrap();
    let squared_diff = diff.mul(&diff).unwrap();
    squared_diff.sum().unwrap().get_data()[0] / (predicted.get_data().len() as f64)
}

fn mse_derivative<T: Tensor<f64>>(predicted: &T, actual: &T) -> impl Tensor<f64> {
    let diff = predicted.sub(actual).unwrap();
    diff.scale(2.0 / (predicted.get_data().len() as f64))
        .unwrap()
}

pub struct NeuralNetwork<T: Tensor<f64>> {
    layers: Vec<Box<dyn Layer<T>>>,
    loss: fn(&T, &T) -> f64,
    loss_derivative: fn(&T, &T) -> T,
}

impl<T: Tensor<f64>> NeuralNetwork<T> {
    pub fn new(loss: fn(&T, &T) -> f64, loss_derivative: fn(&T, &T) -> T) -> Self {
        NeuralNetwork {
            layers: Vec::new(),
            loss,
            loss_derivative,
        }
    }

    pub fn add_layer(&mut self, layer: Box<dyn Layer<T>>) {
        self.layers.push(layer);
    }

    pub fn forward(&mut self, input: T) -> T {
        let mut output = input;
        for layer in self.layers.iter_mut() {
            output = layer.forward(output);
        }
        output
    }

    pub fn backward(&mut self, predicted: &T, actual: &T, learning_rate: f64) {
        let mut error = (self.loss_derivative)(predicted, actual);
        for layer in self.layers.iter_mut().rev() {
            error = layer.backward(&error, learning_rate);
        }
    }

    pub fn train(&mut self, x_train: &T, y_train: &T, epochs: u32, learning_rate: f64) {
        for i in 0..epochs {
            let predicted = self.forward(x_train.clone());

            self.backward(&predicted, y_train, learning_rate);
        }
    }
}

fn build_neural_net<T: Tensor<f64> + 'static>(features: u32, output_size: u32) -> NeuralNetwork<T> {
    let mut nn = NeuralNetwork::new(log_loss, log_loss_derivative);

    nn.add_layer(Box::new(LinearLayer::new(features, 21)));
    nn.add_layer(Box::new(ActivationLayer::new(relu, relu_derivative)));

    nn.add_layer(Box::new(LinearLayer::new(21, 21)));
    nn.add_layer(Box::new(ActivationLayer::new(relu, relu_derivative)));

    nn.add_layer(Box::new(LinearLayer::new(21, 6)));
    nn.add_layer(Box::new(ActivationLayer::new(relu, relu_derivative)));

    nn.add_layer(Box::new(LinearLayer::new(6, output_size)));
    nn.add_layer(Box::new(ActivationLayer::new(sigmoid, sigmoid_prime)));

    nn
}

pub fn run_neural_network<T: Tensor<f64> + 'static>() {
    // Placeholder for loading data
    let Data {
        neural_network: xy, ..
    } = crate::read_file::deserialize_data("data.json").unwrap();

    let x_train = T::new(vec![xy.m, xy.n], xy.x.clone()).unwrap();
    let y_train = T::new(vec![xy.m, 1], xy.y.clone()).unwrap();

    // let (x_train, x_mean, x_std) = normalize_features_mean_std(&x_train);

    // let (y_train, y_mean, y_std) = normalize_features_mean_std(&y_train);

    let epochs = 5000;
    let learning_rate = 0.01;

    let input_size = x_train.get_shape()[1];
    let output_size = y_train.get_shape()[1];

    let mut nn = build_neural_net(input_size, output_size);

    nn.train(&x_train, &y_train, epochs, learning_rate);

    // Initialize test data (the linear_regression function will handle normalization and bias)
    let x_test = T::new(vec![xy.m_test, xy.n], xy.x_test.clone()).unwrap();
    let y_test = T::new(vec![xy.m_test, 1], xy.y_test.clone()).unwrap();

    // let x_test = normalize_features(&x_test, &x_mean, &x_std);

    // Make predictions using the trained weights
    let predictions = nn.forward(x_test);

    // Denormalize predictions
    // let predictions = denormalize_features(&predictions, &y_mean, &y_std);

    // Calculate Mean Squared Error
    let mut total_squared_error = 0.0;
    let total = xy.m_test as usize;

    for i in 0..total {
        let pred = predictions.get_data()[i];
        let actual = y_test.get_data()[i];
        println!(
            "Predicted: {:.4}, Actual: {:.4}, {}",
            pred,
            actual,
            if (pred - actual).abs() < 0.5 {
                "✓"
            } else {
                "✗"
            }
        );
        let error = pred - actual;
        total_squared_error += error * error;
    }

    let mse = total_squared_error / (total as f64);
    println!("\nResults:");
    println!("Total test samples: {}", total);
    println!("Mean Squared Error: {:.4}", mse);
    println!("Root MSE: {:.4}", mse.sqrt() as f64);
}
