use crate::tensor::math::TensorMath;
use crate::tensor::Tensor;

use crate::commons::add_bias_term;

/// Perform a single gradient descent update step.
///
/// - `x`: input features (may include bias column if already added)
/// - `y`: target outputs
/// - `w`: current weights
/// - `l`: learning rate
/// - `logistic`: whether to use a logistic (sigmoid) activation
pub fn gradient_descent<T>(x: &T, y: &T, w: &T, l: f64, logistic: bool) -> Result<T, String>
where
    T: Tensor<f64> + TensorMath<f64, MathOutput = T>,
{
    let data_size = *(x.get_shape().first().ok_or("X must have a shape")?) as f64;

    let lines = x.mul(w)?;

    let prediction = match logistic {
        true => lines.sigmoid(),
        false => Ok(lines),
    }?;

    let loss = prediction.sub(y)?;

    let gradient_raw = x.t()?.mul(&loss)?;

    let d = gradient_raw.scale(l / data_size)?;

    w.sub(&d)
}

/// Train a linear regression model using gradient descent.
///
/// - `x`, `y`: training data
/// - `w`: initial weights (consumed)
/// - `l`: learning rate
/// - `e`: number of epochs
pub fn linear_regression<T>(x: &T, y: &T, w: T, l: f64, e: u32) -> Result<T, String>
where
    T: Tensor<f64> + TensorMath<f64, MathOutput = T>,
{
    let x_with_bias = add_bias_term(x)?;
    let mut weight = w;
    for _ in 0..(e - 1) {
        weight = gradient_descent(&x_with_bias, y, &weight, l, false).unwrap();
    }

    Ok(gradient_descent(&x_with_bias, y, &weight, l, false).unwrap())
}

/// Train a logistic regression model using gradient descent.
///
/// Same parameters as `linear_regression`, but applies a sigmoid
/// activation during training.
pub fn logistic_regression<T>(x: &T, y: &T, w: T, l: f64, e: u32) -> Result<T, String>
where
    T: Tensor<f64> + TensorMath<f64, MathOutput = T>,
{
    let x_with_bias = add_bias_term(x)?;
    let mut weight = w;

    for _ in 0..(e - 1) {
        weight = gradient_descent(&x_with_bias, y, &weight, l, true).unwrap();
    }

    Ok(gradient_descent(&x_with_bias, y, &weight, l, true).unwrap())
}

/// Predict outputs for `x` using linear model weights `w`.
pub fn predict_linear<T>(x: &T, w: &T) -> Result<T, String>
where
    T: Tensor<f64> + TensorMath<f64, MathOutput = T>,
{
    x.mul(w)
}

/// Predict binary labels for `x` using logistic model weights `w`.
///
/// Returns a tensor of predicted labels (0.0 or 1.0).
pub fn predict_logistic<T>(x: &T, w: &T) -> Result<T, String>
where
    T: Tensor<f64> + TensorMath<f64, MathOutput = T>,
{
    let x_with_bias = add_bias_term(x)?;

    let z = x_with_bias.mul(w)?;

    let probabilities = z.sigmoid()?;

    let shape = probabilities.get_shape().clone();
    let predictions_data = probabilities
        .get_data()
        .iter()
        .map(|&p| if p >= 0.5 { 1.0 } else { 0.0 })
        .collect();

    T::new(shape, predictions_data)
}
