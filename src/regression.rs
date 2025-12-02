//! # Regression Module - High-Level Training Interface
//!
//! Provides orchestrated training and evaluation functions for both linear and logistic regression.
//!
//! This module offers a higher-level abstraction over the core gradient descent functions,
//! handling data management, training loops, and performance evaluation automatically.
//!
//! ## Features
//!
//! - **Data Management**: Structures for organizing train/test splits
//! - **Training Orchestration**: Complete training loops with progress reporting
//! - **Performance Metrics**:
//!   - Logistic regression: Classification accuracy
//!   - Linear regression: Mean squared error (MSE)
//! - **Input Flexibility**: JSON deserialization for data loading
//! - **Validation**: Sanity checks on data dimensions

use crate::{linear_regression, logistic_regression, predict_linear, predict_logistic};
use crate::{Tensor, GLOBAL_CONTEXT};
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Training and test data container for regression tasks
///
/// Organizes feature and label data for both training and testing phases.
///
/// # Fields
///
/// * `m` - Number of training samples
/// * `n` - Number of features per sample
/// * `x` - Training feature matrix (length: m × n), stored row-major
/// * `y` - Training labels (length: m)
/// * `m_test` - Number of test samples
/// * `x_test` - Test feature matrix (length: m_test × n)
/// * `y_test` - Test labels (length: m_test)
///
/// # JSON Format
///
/// ```json
/// {
///   "m": 100,
///   "n": 5,
///   "x": [1.0, 2.0, ...],
///   "y": [0.0, 1.0, ...],
///   "m_test": 20,
///   "x_test": [...],
///   "y_test": [...]
/// }
/// ```
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct XY {
    pub m: u32,           // Number of data points
    pub n: u32,           // Number of features
    pub x: Vec<f64>,      // A matrix of data length m * n
    pub y: Vec<f64>,      // A matrix of data length m
    pub m_test: u32,      // Number of test data points
    pub x_test: Vec<f64>, // A matrix of test data length m_test * n
    pub y_test: Vec<f64>, // A matrix of test data length m_test
}

/// Container for complete dataset with both linear and logistic regression splits
///
/// Used for loading datasets configured for multiple regression tasks from JSON.
///
/// # Fields
///
/// * `linear` - Data configuration for linear regression
/// * `logistic` - Data configuration for logistic regression (binary classification)
/// * `neural_network` - Data configuration for neural network (can be any function)
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Data {
    pub linear: XY,
    pub logistic: XY,
    pub neural_network: XY,
}

/// Trains and evaluates logistic regression model
///
/// High-level training orchestration for binary classification. Automates:
/// - Training loop with specified learning rate and epochs
/// - Feature normalization and bias handling (automatic)
/// - Test set evaluation with accuracy metrics
/// - Progress reporting every 10 iterations
///
/// # Arguments
///
/// * `xy` - Data container with training and test sets
/// * `l` - Learning rate controlling gradient descent step size
/// * `e` - Number of training epochs
///
/// # Output
///
/// Prints:
/// - Dataset dimensions for verification
/// - Iteration progress (every 10 steps)
/// - Total training time
/// - Classification accuracy on test set
///
/// # Example
///
/// ```rust,no_run
/// use iron_learn::{run_logistic};
///
/// run_logistic();  // Learning rate 0.01, 10k iterations
/// ```
pub fn run_logistic() {
    let l = GLOBAL_CONTEXT.get().unwrap().learning_rate;
    let e = GLOBAL_CONTEXT.get().unwrap().epochs;
    let data_path = &GLOBAL_CONTEXT.get().unwrap().data_path;

    let Data { logistic: xy, .. } = crate::read_file::deserialize_data(&data_path).unwrap();

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
    let iter10 = Instant::now();

    for i in 0..e {
        if i % 10 == 0 {
            print!("Iteration: {} took {:.2?} \n", i, iter10.elapsed());
        }

        w = logistic_regression(&x, &y, &w, l)
    }
    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);

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

/// Trains and evaluates linear regression model
///
/// High-level training orchestration for continuous value prediction. Automates:
/// - Training loop with specified learning rate and epochs
/// - Feature normalization and bias handling (automatic)
/// - Test set evaluation with MSE metrics
/// - Graceful handling of missing test data
///
/// # Arguments
///
/// * `xy` - Data container with training and optional test sets
/// * `l` - Learning rate controlling gradient descent step size
/// * `e` - Number of training epochs
///
/// # Output
///
/// Prints:
/// - Dataset dimensions for verification
/// - Total training time
/// - Mean squared error (MSE) and root MSE on test set (if available)
/// - Message if no test data is provided
///
/// # Example
///
/// ```rust,no_run
/// use iron_learn::run_linear;
///
/// run_linear();  // Learning rate 0.01, 10k iterations
/// ```
pub fn run_linear() {
    let l = GLOBAL_CONTEXT.get().unwrap().learning_rate;
    let e = GLOBAL_CONTEXT.get().unwrap().epochs;
    let data_path = &GLOBAL_CONTEXT.get().unwrap().data_path;

    let Data { linear: xy, .. } = crate::read_file::deserialize_data(data_path).unwrap();
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
    println!("Root MSE: {:.4}", mse.sqrt() as f64);
}
