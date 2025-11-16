use cust::prelude::*;
use iron_learn::Tensor;
use iron_learn::{linear_regression, logistic_regression, predict_linear, predict_logistic};
use serde::{Deserialize, Serialize};
use std::fs;
use std::time::Instant;

#[derive(Debug, Deserialize, Serialize)]
struct XY {
    m: u32,           // Number of data points
    n: u32,           // Number of features
    x: Vec<f64>,      // A matrix of data length m * n
    y: Vec<f64>,      // A matrix of data length m
    m_test: u32,      // Number of test data points
    x_test: Vec<f64>, // A matrix of test data length m_test * n
    y_test: Vec<f64>, // A matrix of test data length m_test
}

#[derive(Debug, Deserialize, Serialize)]
struct Data {
    linear: XY,
    logistic: XY,
}

// Extracted function: runs the logistic regression flow that was previously in `main`
fn run_logistic(xy: &XY, l: f64, e: u32) {
    print!("\nLogistic Regression\n");
    print!("Number of examples (m): {}\n", xy.m);
    print!("Number of features (n): {}\n", xy.n);
    print!("Total X values length: {}\n", xy.x.len());
    print!("Total Y values length: {}\n", xy.y.len());

    // Sanity checks: ensure x and y lengths match m and n
    let expected_x_len = (xy.m * xy.n) as usize;
    if xy.x.len() != expected_x_len {
        eprintln!(
            "Error: xy.x.len() = {}, but expected m * n = {}. Aborting logistic run.",
            xy.x.len(),
            expected_x_len
        );
        return;
    }
    if xy.y.len() != xy.m as usize {
        eprintln!(
            "Error: xy.y.len() = {}, but expected m = {}. Aborting logistic run.",
            xy.y.len(),
            xy.m
        );
        return;
    }

    let x = Tensor::new(vec![xy.m, xy.n], xy.x.clone()).unwrap();
    let y = Tensor::new(vec![xy.m, 1], xy.y.clone()).unwrap();

    // Initialize weights including bias term (one extra weight)
    let mut w = Tensor::new(vec![xy.n + 1, 1], vec![0.0; (xy.n + 1) as usize]).unwrap();

    let now = Instant::now();

    for _ in 0..e {
        w = logistic_regression(&x, &y, &w, l)
    }
    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);

    println!("Final weights: {}", w);

    // Initialize test data (no need to add bias here, predict_logistic will handle that)
    let x_test = Tensor::new(vec![xy.m_test, xy.n], xy.x_test.clone()).unwrap();
    let y_test = Tensor::new(vec![xy.m_test, 1], xy.y_test.clone()).unwrap();

    // Make predictions (predict_logistic will add bias term internally)
    let predictions = predict_logistic(&x_test, &w);

    // Calculate accuracy
    let mut correct = 0;
    let total = xy.m_test as usize;

    for i in 0..total {
        let pred = predictions.get_data()[i];
        let actual = y_test.get_data()[i];
        if (pred - actual).abs() < 1e-10 {
            // Using small epsilon for floating point comparison
            correct += 1;
        }
    }

    let accuracy = (correct as f64) / (total as f64) * 100.0;
    println!("\nResults:");
    println!("Total samples: {}", total);
    println!("Correct predictions: {}", correct);
    println!("Accuracy: {:.2}%", accuracy);
}

// Extracted function: runs the linear regression flow that was previously in `main`
fn run_linear(xy: &XY, l: f64, e: u32) {
    print!("\nLinear Regression\n");
    print!("Number of examples (m): {}\n", xy.m);
    print!("Number of features (n): {}\n", xy.n);
    print!("Total X values length: {}\n", xy.x.len());
    print!("Total Y values length: {}\n", xy.y.len());

    // Sanity checks: ensure x and y lengths match m and n
    let expected_x_len = (xy.m * xy.n) as usize;
    if xy.x.len() != expected_x_len {
        eprintln!(
            "Error: xy.x.len() = {}, but expected m * n = {}. Aborting linear run.",
            xy.x.len(),
            expected_x_len
        );
        return;
    }
    if xy.y.len() != xy.m as usize {
        eprintln!(
            "Error: xy.y.len() = {}, but expected m = {}. Aborting linear run.",
            xy.y.len(),
            xy.m
        );
        return;
    }

    let x = Tensor::new(vec![xy.m, xy.n], xy.x.clone()).unwrap();
    let y = Tensor::new(vec![xy.m, 1], xy.y.clone()).unwrap();

    // Initialize weights (include bias term: n + 1)
    let mut w = Tensor::new(vec![xy.n + 1, 1], vec![0.0; (xy.n + 1) as usize]).unwrap();

    let now = Instant::now();

    for _ in 0..e {
        w = linear_regression(&x, &y, &w, l)
    }

    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);
    println!("Final weights: {}", w);

    // Skip prediction if there's no test data
    if xy.m_test == 0 {
        println!("\nNo test data available for prediction.");
        return;
    }

    // Initialize test data (the linear_regression function will handle normalization and bias)
    let x_test = Tensor::new(vec![xy.m_test, xy.n], xy.x_test.clone()).unwrap();
    let y_test = Tensor::new(vec![xy.m_test, 1], xy.y_test.clone()).unwrap();

    // Make predictions using the trained weights
    let predictions = predict_linear(&x_test, &w);

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
    println!("Root MSE: {:.4}", mse.sqrt());
}
fn main() {
    let _ctx = cust::quick_init();

    let l = 0.001;

    let e = 10;

    let contents = fs::read_to_string("data.json").expect("Should have been able to read the file");

    // Parse once and destructure into `linear` and `logistic` datasets
    let data: Data = serde_json::from_str(&contents).unwrap();
    let Data {
        linear: xy,
        logistic,
    } = data;

    // Run linear regression using the extracted function
    run_linear(&xy, l, e);

    // Call the extracted function to run logistic regression using the `logistic` dataset
    run_logistic(&logistic, l, e);
}
