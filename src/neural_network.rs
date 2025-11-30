use crate::{tensor::Tensor, Numeric};

pub trait Layer {
    fn forward(&mut self, input: Tensor<f64>) -> Tensor<f64>;
    fn backward(&mut self, error: &Tensor<f64>, learning_rate: f64) -> Tensor<f64>;
}

struct LinearLayer {
    weights: Tensor<f64>,
    bias: Tensor<f64>,
    input: Option<Tensor<f64>>,
}

impl LinearLayer {
    fn new(input_size: u32, output_size: u32) -> Self {
        // Weights shape: [input_size, output_size]
        // This allows: input [batch_size, input_size] × weights [input_size, output_size] = output [batch_size, output_size]
        let weights = Tensor::new(
            vec![input_size, output_size],
            vec![0.0; input_size as usize * output_size as usize],
        )
        .unwrap();
        let bias = Tensor::new(vec![1, output_size], vec![0.0; output_size as usize]).unwrap();
        LinearLayer {
            weights,
            bias,
            input: None,
        }
    }
}

impl Layer for LinearLayer {
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
        for _ in 0..epochs {
            let predicted = self.forward(x_train.clone());
            self.backward(&predicted, y_train, learning_rate);
        }
    }
}

fn build_neural_net(input_size: u32, hidden_size: u32, output_size: u32) -> NeuralNetwork {
    let mut nn = NeuralNetwork::new(mse, mse_derivative);
    nn.add_layer(Box::new(LinearLayer::new(input_size, hidden_size)));
    nn.add_layer(Box::new(ActivationLayer::new(relu, relu_derivative)));
    nn.add_layer(Box::new(LinearLayer::new(hidden_size, output_size)));
    nn
}

pub fn run_neural_network() {
    // Placeholder for loading data
    let x_train = Tensor::new(vec![100, 3], vec![0.0; 300]).unwrap(); // 100 samples, 3 features
    let y_train = Tensor::new(vec![100, 1], vec![0.0; 100]).unwrap(); // 100 samples, 1 target

    let epochs = 1000;
    let learning_rate = 0.01;

    let input_size = x_train.get_shape()[1];
    let hidden_size = 16;
    let output_size = y_train.get_shape()[1];

    let mut nn = build_neural_net(input_size, hidden_size, output_size);

    nn.train(&x_train, &y_train, epochs, learning_rate);

    println!("Training completed.");
    println!("Neural network structure:");
}
