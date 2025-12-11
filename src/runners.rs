use crate::commons::add_bias_term;
use crate::commons::denormalize_features;
use crate::commons::normalize_features_mean_std;
use crate::gpu_ml::activations::sigmoid_activation;
use crate::gpu_ml::activations::sigmoid_derivative;
use crate::gpu_ml::activations::tanh_activation;
use crate::gpu_ml::activations::tanh_derivative;
use crate::gpu_ml::add_bias_column;
use crate::gpu_ml::builder::GpuNetworkBuilder;
use crate::gpu_ml::compute_norm_stats;
use crate::gpu_ml::trainer::GpuNetworkTrainer;
use crate::gpu_ml::trainer::TrainerConfig;
use crate::normalize_features;
use crate::tensor::math::TensorMath;
use crate::tensor::Tensor;
use crate::Numeric;
use crate::{linear_regression, logistic_regression, predict_linear, predict_logistic};
use crate::{CpuTensor, GLOBAL_CONTEXT};
use serde::{Deserialize, Serialize};
use std::time::Instant;

use crate::read_file::deserialize_data;
use cust::prelude::Module;
use cust::stream::Stream;
use cust::stream::StreamFlags;

use crate::ActivationType;
use crate::MeanSquaredErrorLoss;
use crate::NeuralNet;
use crate::NeuralNetBuilder;

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct XY {
    pub m: u32,
    pub n: u32,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub m_test: u32,
    pub x_test: Vec<f64>,
    pub y_test: Vec<f64>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Data {
    pub linear: XY,
    pub logistic: XY,
    pub neural_network: XY,
    pub cat_image: XY,
}

pub fn run_logistic<T>() -> Result<(), String>
where
    T: Tensor<f64> + TensorMath<f64, MathOutput = T>,
{
    let l = GLOBAL_CONTEXT
        .get()
        .ok_or("GLOBAL_CONTEXT not initialized")?
        .learning_rate;
    let e = GLOBAL_CONTEXT.get().unwrap().epochs;
    let data_path = &GLOBAL_CONTEXT.get().unwrap().data_path;

    let Data { logistic: xy, .. } = crate::read_file::deserialize_data(&data_path)
        .map_err(|e| format!("Data deserialization error: {}", e))?;

    print!("\nLogistic Regression\n");
    print!("Number of examples (m): {}\n", xy.m);
    print!("Number of features (n): {}\n", xy.n);
    print!("Total X values length: {}\n", xy.x.len());
    print!("Total Y values length: {}\n", xy.y.len());

    let x = T::new(vec![xy.m, xy.n], xy.x.clone())?;
    let y = T::new(vec![xy.m, 1], xy.y.clone())?;

    let mut w = T::new(vec![xy.n + 1, 1], vec![0.0; (xy.n + 1) as usize])?;

    let now = Instant::now();
    let iter10 = Instant::now();

    w = logistic_regression(&x, &y, w, l, e).unwrap();
    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);

    let x_test = T::new(vec![xy.m_test, xy.n], xy.x_test.clone())?;
    let y_test = T::new(vec![xy.m_test, 1], xy.y_test.clone())?;

    let predictions = predict_logistic(&x_test, &w)?;

    let mut correct = 0;
    let total = xy.m_test as usize;
    let predictions_data = predictions.get_data();
    let y_test_data = y_test.get_data();

    for i in 0..total {
        let pred = predictions_data[i];
        let actual = y_test_data[i];
        if (pred - actual).abs() < 1e-10 {
            correct += 1;
        }
    }

    let accuracy = (correct as f64) / (total as f64) * 100.0;
    println!("\nResults:");
    println!("Total samples: {}", total);
    println!("Correct predictions: {}", correct);
    println!("Accuracy: {:.2}%", accuracy);

    Ok(())
}

pub fn run_linear<T>() -> Result<(), String>
where
    T: Tensor<f64> + TensorMath<f64, MathOutput = T>,
{
    let l = GLOBAL_CONTEXT
        .get()
        .ok_or("GLOBAL_CONTEXT not initialized")?
        .learning_rate;
    let e = GLOBAL_CONTEXT.get().unwrap().epochs;
    let data_path = &GLOBAL_CONTEXT.get().unwrap().data_path;

    let Data { linear: xy, .. } = crate::read_file::deserialize_data(data_path)
        .map_err(|e| format!("Data deserialization error: {}", e))?;

    print!("\nLinear Regression\n");
    print!("Number of examples (m): {}\n", xy.m);
    print!("Number of features (n): {}\n", xy.n);
    print!("Total X values length: {}\n", xy.x.len());
    print!("Total Y values length: {}\n", xy.y.len());

    let x = T::new(vec![xy.m, xy.n], xy.x.clone())?;
    let y = T::new(vec![xy.m, 1], xy.y.clone())?;

    let (x, x_mean, x_std) = normalize_features_mean_std(&x);
    let (y, y_mean, y_std) = normalize_features_mean_std(&y);

    let mut w = T::new(vec![xy.n + 1, 1], vec![0.0; (xy.n + 1) as usize])?;

    let now = Instant::now();

    w = linear_regression(&x, &y, w, l, e).unwrap();

    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);

    if xy.m_test == 0 {
        println!("\nNo test data available for prediction.");
        return Ok(());
    }

    let x_test = T::new(vec![xy.m_test, xy.n], xy.x_test.clone())?;
    let y_test = T::new(vec![xy.m_test, 1], xy.y_test.clone())?;

    let x_test = normalize_features(&x_test, &x_mean, &x_std);

    let predictions = predict_linear(&x_test, &w)?;

    let predictions = denormalize_features(&predictions, &y_mean, &y_std);

    let mut total_squared_error = 0.0;
    let total = xy.m_test as usize;
    let predictions_data = predictions.get_data();
    let y_test_data = y_test.get_data();

    for i in 0..total {
        let pred = predictions_data[i];
        let actual = y_test_data[i];
        let error = pred - actual;
        total_squared_error += error * error;
    }

    let mse = total_squared_error / (total as f64);
    println!("\nResults:");
    println!("Total test samples: {}", total);
    println!("Mean Squared Error: {:.4}", mse);
    println!("Root MSE: {:.4}", mse.sqrt());

    Ok(())
}

pub fn run_neural_net<T>() -> Result<(), String>
where
    T: Tensor<f64> + TensorMath<f64, MathOutput = T> + 'static,
{
    let l = GLOBAL_CONTEXT
        .get()
        .ok_or("GLOBAL_CONTEXT not initialized")?
        .learning_rate;
    let e = GLOBAL_CONTEXT.get().unwrap().epochs;
    let data_path = &GLOBAL_CONTEXT.get().unwrap().data_path;

    let Data {
        neural_network: xy, ..
    } = crate::read_file::deserialize_data(data_path)
        .map_err(|e| format!("Data deserialization error: {}", e))?;

    let monitor = |epoch: usize, err: f64| {
        // This line runs on the Host (CPU) but the error value came from a D2H transfer
        if epoch % 500 == 0 || epoch == (e - 1) as usize {
            println!("\tEpoch {}: Loss (MSE) = {:.8}", epoch, err / 4.0); // Divide by 4 samples
        }
    };

    let x = T::new(vec![xy.m, xy.n], xy.x.clone())?;
    let y = T::new(vec![xy.m, 1], xy.y.clone())?;

    let (x, x_mean, x_std) = normalize_features_mean_std(&x);
    let (y, y_mean, y_std) = normalize_features_mean_std(&y);

    let x = add_bias_term(&x).unwrap();

    let mut w = T::new(vec![xy.n, 1], vec![0.0; xy.n as usize])?;

    let loss_function_instance = Box::new(MeanSquaredErrorLoss);
    let hidden_length = 6;
    let input_length = 2;

    let nn = NeuralNetBuilder::<T>::new();
    let mut nn = nn
        .add_linear(input_length + 1, hidden_length, "Input")
        .add_activation(ActivationType::Tanh, "Activatin Layer 1")
        .add_linear(hidden_length, hidden_length, "Hidden Layer 1")
        .add_activation(ActivationType::Tanh, "Activation Layer 2")
        .add_linear(hidden_length, hidden_length, "Hidden Layer 2")
        .add_activation(ActivationType::Tanh, "Activation Layer 3")
        .add_linear(hidden_length, hidden_length / 2, "Hidden Layer 3")
        .add_activation(ActivationType::Tanh, "Activation Layer 4")
        .add_linear(hidden_length / 2, hidden_length / 2, "Hidden Layer 4")
        .add_activation(ActivationType::Tanh, "Activation Layer 5")
        .add_linear(hidden_length / 2, hidden_length / 2, "Hidden Layer 5")
        .add_activation(ActivationType::Tanh, "Activation Layer 6")
        .add_linear(hidden_length / 2, 1, "Hidden Layer 6")
        .add_activation(ActivationType::Tanh, "Activation Layer 7")
        .build(loss_function_instance);

    nn.fit(&x, &y, 10000, 0, 0.1, monitor);

    let x_test = T::new(vec![xy.m_test, xy.n], xy.x_test.clone())?;
    let y_test = T::new(vec![xy.m_test, 1], xy.y_test.clone())?;

    println!("{} {}", xy.m_test, xy.n);

    let x_test = normalize_features(&x_test, &x_mean, &x_std);

    println!("{:?}", x_test.get_shape());

    println!("{:?}", x_test.get_data());

    let predictions = predict_linear(&x_test, &w).expect("Did not work");

    println!("{:?}", predictions.get_data());

    let predictions = denormalize_features(&predictions, &y_mean, &y_std);

    for i in 0..y_test.get_data().len() {
        println!(
            "Y: {}, P: {}",
            y_test.get_data()[i],
            predictions.get_data()[i]
        );
    }

    Ok(())
}

pub fn run_custom_network() {
    let learning_rate = GLOBAL_CONTEXT.get().unwrap().learning_rate;
    let epochs = GLOBAL_CONTEXT.get().unwrap().epochs;
    let data_path = &GLOBAL_CONTEXT.get().unwrap().data_path;

    let Data {
        neural_network: data,
        ..
    } = deserialize_data(data_path).unwrap();
    let rows = data.m as usize;
    let cols = data.n as usize;
    let input_cols = cols + 1;

    let x_bias = add_bias_column(&data.x, rows, cols);

    let ptx = include_str!("../kernels/gpu_kernels.ptx");
    let module = Module::from_ptx(ptx, &[]).unwrap();
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

    let hidden_length = 4;

    let mut network = GpuNetworkBuilder::new();


    network.add_linear("InputLayer", input_cols, hidden_length);
    network.add_activation("TanH 1", tanh_activation, tanh_derivative);

    network.add_linear("HiddenLayer2", hidden_length, hidden_length);
    network.add_activation("TanH 2", tanh_activation, tanh_derivative);

    network.add_linear("HiddenLayer3", hidden_length, hidden_length / 2);
    network.add_activation("TanH 3", tanh_activation, tanh_derivative);

    network.add_linear("OutputLayer", hidden_length / 2, 1);
    network.add_activation("Sigmoid", sigmoid_activation, sigmoid_derivative);

    let network = network.build();

    network.print_architecture();

    let config = TrainerConfig {
        learning_rate: learning_rate/10.0, // Use lower learning rate for this config
        epochs: epochs as usize,
        checkpoint_interval: 2000,
        hidden_size: 6,
    };

    let mut trainer = GpuNetworkTrainer::new(network, config, &module, stream);
    let (duration, final_loss) = trainer
        .fit(&x_bias, &data.y, rows as usize, input_cols as usize)
        .unwrap();

    let x_test = data.x_test;
    let x_test = add_bias_column(&x_test, data.m_test as usize, data.n as usize);
    let y_test = data.y_test;
    let m_test = data.m_test;

    let ev = trainer.evaluate(&x_test, &y_test, m_test as usize, data.n as usize);

    println!("\nTraining Results:");
    println!("  Time: {:?}", duration);
    println!("  Final Loss: {}", final_loss);
}
