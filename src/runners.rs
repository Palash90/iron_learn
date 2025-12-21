// use crate::commons::add_bias_term;
use crate::commons::denormalize_features;
use crate::commons::normalize_features_mean_std;
use crate::neural_network::{sigmoid, sigmoid_prime, tanh, tanh_prime};
use crate::normalize_features;
use crate::read_file::deserialize_data_double_precision;
use crate::tensor::math::TensorMath;
use crate::tensor::Tensor;
use crate::GLOBAL_CONTEXT;
use crate::{linear_regression, logistic_regression, predict_linear, predict_logistic};
use serde::{Deserialize, Serialize};
use std::time::Instant;

use crate::read_file::deserialize_data;

use crate::neural_network::NeuralNetDataType;
use crate::NeuralNet;
use crate::NeuralNetBuilder;

use crate::neural_network::MeanSquaredErrorLoss;

use crate::commons::add_bias_term;

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct XY {
    pub m: u32,
    pub n: u32,
    pub x: Vec<NeuralNetDataType>,
    pub y: Vec<NeuralNetDataType>,
    pub m_test: u32,
    pub x_test: Vec<NeuralNetDataType>,
    pub y_test: Vec<NeuralNetDataType>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Data {
    pub linear: XY,
    pub logistic: XY,
    pub neural_network: XY,
    pub cat_image: XY,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct XYDoublePrecision {
    pub m: u32,
    pub n: u32,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub m_test: u32,
    pub x_test: Vec<f64>,
    pub y_test: Vec<f64>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct DataDoublePrecision {
    pub linear: XYDoublePrecision,
    pub logistic: XYDoublePrecision,
    pub neural_network: XYDoublePrecision,
    pub cat_image: XYDoublePrecision,
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

    let DataDoublePrecision { logistic: xy, .. } = deserialize_data_double_precision(&data_path)
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
    println!("Accuracy: {:.2?}%", accuracy);

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

    let DataDoublePrecision { linear: xy, .. } = deserialize_data_double_precision(data_path)
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
    T: Tensor<NeuralNetDataType> + TensorMath<NeuralNetDataType, MathOutput = T> + 'static,
{
    let l = GLOBAL_CONTEXT
        .get()
        .ok_or("GLOBAL_CONTEXT not initialized")?
        .learning_rate as NeuralNetDataType;
    let e = GLOBAL_CONTEXT.get().unwrap().epochs;
    let data_path = &GLOBAL_CONTEXT.get().unwrap().data_path;

    let Data {
        neural_network: xy, ..
    } = deserialize_data(data_path).map_err(|e| format!("Data deserialization error: {}", e))?;

    let x = T::new(vec![xy.m, xy.n], xy.x.clone())?;
    let y = T::new(vec![xy.m, 1], xy.y.clone())?;

    //let (x, x_mean, x_std) = normalize_features_mean_std(&x);
    //let (y, y_mean, y_std) = normalize_features_mean_std(&y);

    let x = add_bias_term(&x).unwrap();

    let loss_function_instance = Box::new(MeanSquaredErrorLoss);
    let hidden_length = 50; // Should be even
    let input_length = 2;

    let input_length = input_length + 1; // To compensate for bias

    let nn = NeuralNetBuilder::<T>::new();
    let mut nn = nn;

    nn.add_linear(input_length, hidden_length, "Input");
    nn.add_activation(tanh, tanh_prime, "Activation Layer 1");

    nn.add_linear(hidden_length, hidden_length, "Hidden Layer 1");
    nn.add_activation(tanh, tanh_prime, "Activation Layer 2");

    nn.add_linear(hidden_length, 2 * hidden_length, "Hidden Layer 2");
    nn.add_activation(tanh, tanh_prime, "Activation Layer 3");

    nn.add_linear(2 * hidden_length, hidden_length, "Hidden Layer 3");
    nn.add_activation(tanh, tanh_prime, "Activation Layer 4");

    nn.add_linear(hidden_length, hidden_length / 2, "Hidden Layer 4");
    nn.add_activation(tanh, tanh_prime, "Activation Layer 5");

    nn.add_linear(hidden_length / 2, hidden_length / 2, "Hidden Layer 5");
    nn.add_activation(tanh, tanh_prime, "Activation Layer 6");

    nn.add_linear(hidden_length / 2, hidden_length / 2, "Hidden Layer 6");
    nn.add_activation(tanh, tanh_prime, "Activation Layer 7");

    nn.add_linear(hidden_length / 2, 1, "Hidden Layer 7");
    nn.add_activation(sigmoid, sigmoid_prime, "Output Layer");

    let mut nn = nn.build(loss_function_instance);

    let mut start_time = Instant::now();

    let monitor = |epoch: usize, err: NeuralNetDataType, nn: &mut NeuralNet<T>| {
        let elapsed = start_time.elapsed();
        start_time = Instant::now();

        println!("\tEpoch {}: Loss (MSE) = {:.8}", epoch, err);

        println!(
            "Hook completed at epoch {}, time took {:.2?}",
            epoch, elapsed
        );
    };

    let _ = nn.fit(&x, &y, e as usize, 0, l, true, monitor, 1000);

    //let x_test = T::new(vec![xy.m_test, xy.n], xy.x_test.clone())?;

    //let y_test = T::new(vec![xy.m_test, 1], xy.y_test.clone())?;

    //let x_test = normalize_features(&x_test, &x_mean, &x_std);

    //let x_test = add_bias_term(&x_test)?;

    //let predictions = nn.predict(&x).unwrap();

    //predictions.print_matrix();

    //let predictions = denormalize_features(&predictions, &y_mean, &y_std);
    Ok(())
}
