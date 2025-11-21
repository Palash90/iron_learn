//! # Gradient Descent Optimization Module
//!
//! Provides gradient-based optimization algorithms for machine learning tasks.
//!
//! ## Overview
//!
//! This module implements the fundamental gradient descent algorithm with specialized variants
//! for linear and logistic regression. Key features include:
//!
//! - **Feature Normalization**: Z-score standardization for numeric stability
//! - **Bias Handling**: Automatic bias term insertion during preprocessing
//! - **Multiple Regression Modes**: Support for both linear (MSE) and logistic (cross-entropy) objectives
//! - **Flexible API**: Both low-level and high-level interfaces
//!
//! ## Algorithm Flow
//!
//! For each iteration:
//! 1. Compute predictions: z = X · w (linear) or sigmoid(X · w) (logistic)
//! 2. Calculate loss: prediction - target
//! 3. Compute gradient: X^T · loss / m
//! 4. Update weights: w ← w - learning_rate · gradient
//!
//! ## Preprocessing
//!
//! All functions automatically normalize features and add bias terms internally
//! for improved numerical stability and model performance.

use crate::Tensor;

/// Core gradient descent optimization algorithm
///
/// Performs a single step of gradient descent, computing the parameter update based on
/// the current model predictions and loss. This is the foundational optimization primitive
/// used by both linear and logistic regression.
///
/// # Arguments
///
/// * `x` - Input features matrix, shape (m, n) where m = sample count, n = feature count
/// * `y` - Target values vector, shape (m, 1) where m = sample count
/// * `w` - Current weight vector, shape (n, 1) where n = feature count
/// * `l` - Learning rate (step size), controls update magnitude. Typical range: [0.001, 0.1]
/// * `logistic` - Regression mode flag
///   - `true`: Logistic regression (sigmoid activation, binary classification)
///   - `false`: Linear regression (MSE loss, continuous prediction)
///
/// # Returns
///
/// Updated weight vector after one gradient descent step
///
/// # Note
///
/// This is a lower-level function. For typical usage, prefer `linear_regression` or
/// `logistic_regression` which add preprocessing automation.
///
/// # Example
///
/// ```rust
/// use iron_learn::{Tensor, gradient_descent};
/// let x = Tensor::new(vec![3, 2], vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0]).unwrap();
/// let y = Tensor::new(vec![3, 1], vec![5.0, 7.0, 9.0]).unwrap();
/// let w = Tensor::new(vec![2, 1], vec![0.1, 0.2]).unwrap();
/// let updated_w = gradient_descent(&x, &y, &w, 0.01, false);
/// ```
pub fn gradient_descent(
    x: &Tensor<f64>,
    y: &Tensor<f64>,
    w: &Tensor<f64>,
    l: f64,
    logistic: bool,
) -> Tensor<f64> {
    let data_size = *(x.get_shape().first().unwrap()) as f64;
    let lines = x.mul(w).unwrap();

    let prediction = match logistic {
        true => sigmoid(lines),
        false => lines,
    };

    let loss = prediction.sub(&y).unwrap();
    let d = x
        .t()
        .unwrap()
        .mul(&loss) // Multiply `X` to loss
        .unwrap()
        //.add(&(w.scale(lambda/data_size)))// Add Regularization to parameters
        //.unwrap()
        .scale(l / data_size); // Scale gradient by learning rate

    w.sub(&d).unwrap()
}

/// Linear regression with automatic preprocessing
///
/// High-level interface for linear regression that automatically:
/// - Normalizes input features to zero mean and unit variance
/// - Adds bias term to feature matrix
/// - Applies single-step gradient descent with MSE loss
///
/// # Arguments
///
/// * `x` - Input features matrix, shape (m, n) where m = samples, n = features (without bias)
/// * `y` - Target values, shape (m, 1)
/// * `w` - Weight vector, shape (n+1, 1) including bias weight
/// * `l` - Learning rate
///
/// # Returns
///
/// Updated weight vector after one iteration
///
/// # Example
///
/// ```rust
/// use iron_learn::{Tensor, linear_regression};
/// let x = Tensor::new(vec![100, 5], (0..500).map(|i| i as f64).collect()).unwrap();
/// let y = Tensor::new(vec![100, 1], (0..100).map(|i| (i * 2) as f64).collect()).unwrap();
/// let w = Tensor::new(vec![6, 1], vec![0.0; 6]).unwrap();
/// let w = linear_regression(&x, &y, &w, 0.01);
/// ```
pub fn linear_regression(x: &Tensor<f64>, y: &Tensor<f64>, w: &Tensor<f64>, l: f64) -> Tensor<f64> {
    let x_normalized = normalize_features(x);
    let x_with_bias = add_bias_term(&x_normalized);
    gradient_descent(&x_with_bias, y, w, l, false)
}

/// Feature normalization using z-score standardization
///
/// Transforms each feature to have zero mean and unit variance:
/// `z = (x - mean) / std_dev`
///
/// This standardization improves:
/// - Gradient descent convergence speed
/// - Numerical stability
/// - Weight initialization effectiveness
///
/// # Arguments
///
/// * `x` - Input feature matrix, shape (m, n)
///
/// # Returns
///
/// Normalized feature matrix with same shape
fn normalize_features(x: &Tensor<f64>) -> Tensor<f64> {
    let shape = x.get_shape();
    let m = shape[0] as usize;
    let n = shape[1] as usize;
    let mut normalized_data = vec![0.0; m * n];

    // For each feature
    for j in 0..n {
        let mut mean = 0.0;
        let mut variance = 0.0;

        // Calculate mean
        for i in 0..m {
            mean += x.get_data()[i * n + j];
        }
        mean /= m as f64;

        // Calculate variance
        for i in 0..m {
            let diff = x.get_data()[i * n + j] - mean;
            variance += diff * diff;
        }
        variance /= m as f64;
        let std_dev = variance.sqrt();

        // Normalize feature
        for i in 0..m {
            normalized_data[i * n + j] = if std_dev > 1e-8 {
                (x.get_data()[i * n + j] - mean) / std_dev
            } else {
                x.get_data()[i * n + j] - mean
            };
        }
    }

    Tensor::new(shape.clone(), normalized_data).unwrap()
}

/// Bias term insertion for regression models
///
/// Prepends a column of 1.0 values to the feature matrix to represent the intercept term.
/// This enables the model to learn a non-zero mean prediction when all features are zero.
///
/// Transformation: X_with_bias = [1, x₁, x₂, ..., xₙ] for each sample
fn add_bias_term(x: &Tensor<f64>) -> Tensor<f64> {
    let shape = x.get_shape();
    let m = shape[0] as usize; // number of examples
    let n = shape[1] as usize; // number of features
    let mut data = Vec::with_capacity(m * (n + 1));

    // For each example
    for i in 0..m {
        // Add 1 as the bias term
        data.push(1.0);
        // Add the original features
        for j in 0..n {
            data.push(x.get_data()[i * n + j]);
        }
    }

    Tensor::new(vec![shape[0], shape[1] + 1], data).unwrap()
}

/// Logistic regression with automatic preprocessing
///
/// High-level interface for binary classification using logistic regression that automatically:
/// - Normalizes input features
/// - Adds bias term
/// - Applies sigmoid activation function
/// - Uses binary cross-entropy loss
///
/// # Arguments
///
/// * `x` - Input features matrix, shape (m, n)
/// * `y` - Binary target labels, shape (m, 1) with values in {0, 1}
/// * `w` - Weight vector, shape (n+1, 1) including bias weight
/// * `l` - Learning rate
///
/// # Returns
///
/// Updated weight vector after one iteration
///
/// # Example
///
/// ```rust
/// use iron_learn::{Tensor, logistic_regression};
/// let x = Tensor::new(vec![100, 5], (0..500).map(|i| (i % 2) as f64).collect()).unwrap();
/// let y = Tensor::new(vec![100, 1], (0..100).map(|i| (i % 2) as f64).collect()).unwrap();
/// let w = Tensor::new(vec![6, 1], vec![0.0; 6]).unwrap();
/// let w = logistic_regression(&x, &y, &w, 0.01);
/// ```
pub fn logistic_regression(
    x: &Tensor<f64>,
    y: &Tensor<f64>,
    w: &Tensor<f64>,
    l: f64,
) -> Tensor<f64> {
    let x_normalized = normalize_features(x);
    let x_with_bias = add_bias_term(&x_normalized);
    gradient_descent(&x_with_bias, y, w, l, true)
}

/// Sigmoid activation function
///
/// Applies the logistic sigmoid transformation: σ(z) = 1 / (1 + e^(-z))
///
/// This function maps any real value to the range (0, 1), making it suitable
/// for probabilistic interpretations in classification tasks.
///
/// # Arguments
///
/// * `z` - Input tensor
///
/// # Returns
///
/// Output tensor with same shape containing sigmoid-transformed values
fn sigmoid(z: Tensor<f64>) -> Tensor<f64> {
    let result = Tensor::exp(&-z);
    let shape = result.get_shape();
    let result = result.get_data().iter().map(|t| 1.0 / (1.0 + t)).collect();

    Tensor::new(shape, result).unwrap()
}

/// Prediction function for linear regression
///
/// Generates predictions on test data using trained weights.
///
/// Automatically applies the same preprocessing (normalization and bias addition)
/// as was used during training for consistency.
///
/// # Arguments
///
/// * `x` - Test input features, shape (m, n) where m = test samples, n = features
/// * `w` - Trained weight vector, shape (n+1, 1) including bias weight
///
/// # Returns
///
/// Predicted continuous values, shape (m, 1)
///
/// # Example
///
/// ```rust
/// use iron_learn::{Tensor, predict_linear};
/// let x_test = Tensor::new(vec![10, 5], (0..50).map(|i| i as f64).collect()).unwrap();
/// let w = Tensor::new(vec![6, 1], vec![0.5; 6]).unwrap();
/// let predictions = predict_linear(&x_test, &w);
/// ```
pub fn predict_linear(x: &Tensor<f64>, w: &Tensor<f64>) -> Tensor<f64> {
    // First normalize features
    let x_normalized = normalize_features(x);
    // Then add bias term
    let x_with_bias = add_bias_term(&x_normalized);
    x_with_bias.mul(w).unwrap()
}

/// Prediction function for logistic regression
///
/// Generates binary class predictions on test data using trained weights.
///
/// Applies sigmoid activation to the linear predictions, then thresholds at 0.5:
/// - If σ(X · w) ≥ 0.5 → predicts class 1
/// - If σ(X · w) < 0.5 → predicts class 0
///
/// Automatically applies the same preprocessing (normalization and bias addition)
/// as was used during training.
///
/// # Arguments
///
/// * `x` - Test input features, shape (m, n)
/// * `w` - Trained weight vector, shape (n+1, 1) including bias weight
///
/// # Returns
///
/// Binary predictions tensor, shape (m, 1) with values in {0.0, 1.0}
///
/// # Example
///
/// ```rust
/// use iron_learn::{Tensor, predict_logistic};
/// let x_test = Tensor::new(vec![10, 5], (0..50).map(|i| (i % 2) as f64).collect()).unwrap();
/// let w = Tensor::new(vec![6, 1], vec![0.1; 6]).unwrap();
/// let predictions = predict_logistic(&x_test, &w);
/// ```
pub fn predict_logistic(x: &Tensor<f64>, w: &Tensor<f64>) -> Tensor<f64> {
    // First normalize features
    let x_normalized = normalize_features(x);
    // Then add bias term
    let x_with_bias = add_bias_term(&x_normalized);
    let z = x_with_bias.mul(w).unwrap();
    let probabilities = sigmoid(z);
    let shape = probabilities.get_shape();
    let predictions = probabilities
        .get_data()
        .iter()
        .map(|&p| if p >= 0.5 { 1.0 } else { 0.0 })
        .collect();

    Tensor::new(shape, predictions).unwrap()
}
