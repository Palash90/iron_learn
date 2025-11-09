//! The `gradient_descent` module provides gradient descent optimization

use crate::Tensor;

/// The `gradient_descent` function performs a single step of the gradient descent optimization algorithm.
///
/// # Arguments
///
/// * `x` - All values of all features arranged in a m * n matrix, where m is nuber of data points and n is number of features.
/// * `w` - The current value of the weights for each feature, arranged in n * 1 tensor, where n is number of features.
/// * `y` - The collected target values arranged in a m * 1 matrix, where m is the number of data points.
/// * `l` - The learning rate, which controls the size of the update step.
/// * `logistic` - The indicator for Logistic Regression. If set to true, the gradient descent used `Sigmoid` function as classification algorithm.
/// * `lambda` - The regularization parameter to prevent overfitting.
/// # Returns
///
/// The updated weight matrix
///
/// # Example
///
/// ```
/// use iron_learn::Tensor;
/// use iron_learn::gradient_descent::gradient_descent;
///
/// let learning_rate: f64 = 0.01;
/// let w = Tensor::new(vec![2, 1], vec![3.0, 4.0]).unwrap();
/// let x = Tensor::new(vec![1, 2], vec![3.0, 4.0]).unwrap();
/// let y = Tensor::new(vec![1, 1], vec![5.0]).unwrap();
///
/// let w = gradient_descent(&x, &y, &w, learning_rate, false, 10.0); // For linear regression
/// ```
///
///
#[deprecated(since = "0.4.0")]
pub fn gradient_descent(
    x: &Tensor<f64>,
    y: &Tensor<f64>,
    w: &Tensor<f64>,
    l: f64,
    logistic: bool,
    lambda: f64,
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

/// Same as `gradient_descent`, only without the logistic flag parameter. This function invokes `gradient_descent` with the `logistic` flag set to `false`.
pub fn linear_regression(x: &Tensor<f64>, y: &Tensor<f64>, w: &Tensor<f64>, l: f64) -> Tensor<f64> {
    // Normalize features and add bias term for linear regression to improve numeric stability
    let x_normalized = normalize_features(x);
    let x_with_bias = add_bias_term(&x_normalized);
    gradient_descent(&x_with_bias, y, w, l, false, 0.0)
}

/// Same as `gradient_descent`, only without the logistic flag parameter. This function invokes `gradient_descent` with the `logistic` flag set to `true`.
/// Normalizes features to have zero mean and unit variance
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

/// Adds a column of 1s to the input features matrix to handle bias term
fn add_bias_term(x: &Tensor<f64>) -> Tensor<f64> {
    let shape = x.get_shape();
    let m = shape[0] as usize;  // number of examples
    let n = shape[1] as usize;  // number of features
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

pub fn logistic_regression(
    x: &Tensor<f64>,
    y: &Tensor<f64>,
    w: &Tensor<f64>,
    l: f64,
) -> Tensor<f64> {
    // First normalize features
    let x_normalized = normalize_features(x);
    // Then add bias term
    let x_with_bias = add_bias_term(&x_normalized);
    gradient_descent(&x_with_bias, y, w, l, true, 1.0)
}

fn sigmoid(z: Tensor<f64>) -> Tensor<f64> {
    let result = Tensor::exp(&-z);
    let shape = result.get_shape();
    let result = result.get_data().iter().map(|t| 1.0 / (1.0 + t)).collect();

    Tensor::new(shape, result).unwrap()
}

/// Makes predictions for linear regression.
/// 
/// # Arguments
/// 
/// * `x` - The test input features arranged in a m * n matrix, where m is number of test points and n is number of features
/// * `w` - The trained weights arranged in (n+1) * 1 tensor (includes bias weight)
/// 
/// # Returns
/// 
/// A tensor of predicted values
pub fn predict_linear(x: &Tensor<f64>, w: &Tensor<f64>) -> Tensor<f64> {
    // First normalize features
    let x_normalized = normalize_features(x);
    // Then add bias term
    let x_with_bias = add_bias_term(&x_normalized);
    x_with_bias.mul(w).unwrap()
}

/// Predicts binary classes for logistic regression. Returns 1 if the probability is >= 0.5, 0 otherwise.
/// 
/// # Arguments
/// 
/// * `x` - The test input features arranged in a m * n matrix, where m is number of test points and n is number of features
/// * `w` - The trained weights arranged in n * 1 tensor
/// 
/// # Returns
/// 
/// A tensor of binary predictions (0 or 1)
pub fn predict_logistic(x: &Tensor<f64>, w: &Tensor<f64>) -> Tensor<f64> {
    // First normalize features
    let x_normalized = normalize_features(x);
    // Then add bias term
    let x_with_bias = add_bias_term(&x_normalized);
    let z = x_with_bias.mul(w).unwrap();
    let probabilities = sigmoid(z);
    let shape = probabilities.get_shape();
    let predictions = probabilities.get_data().iter().map(|&p| if p >= 0.5 { 1.0 } else { 0.0 }).collect();
    
    Tensor::new(shape, predictions).unwrap()
}
