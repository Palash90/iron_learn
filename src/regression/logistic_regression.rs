use crate::{
    commons::add_bias_term, gradient_descent::gradient_descent, numeric::FloatingPoint,
    tensor::math::TensorMath, Tensor,
};

use std::io;
use std::io::Write;

/// Predict outputs for `x` using linear model weights `w`.
/// Train a logistic regression model using gradient descent.
///
/// Same parameters as `linear_regression`, but applies a sigmoid
/// activation during training.
pub fn logistic_regression<T, D>(x: &T, y: &T, w: T, l: D, e: u32) -> Result<T, String>
where
    T: Tensor<D> + TensorMath<D, MathOutput = T>,
    D: FloatingPoint,
{
    let x_with_bias = add_bias_term(x)?;
    let mut weight = w;

    for i in 0..(e - 1) {
        print!("\rProcessing epoch: {i}/{e}");
        io::stdout().flush().unwrap();

        weight = gradient_descent(&x_with_bias, y, &weight, l, true).unwrap();
    }

    Ok(gradient_descent(&x_with_bias, y, &weight, l, true).unwrap())
}

/// Predict binary labels for `x` using logistic model weights `w`.
///
/// Returns a tensor of predicted labels (0.0 or 1.0).
pub fn predict_logistic<T, D>(x: &T, w: &T) -> Result<T, String>
where
    T: Tensor<D> + TensorMath<D, MathOutput = T>,
    D: FloatingPoint,
{
    let x_with_bias = add_bias_term(x)?;

    let z = x_with_bias.matmul(w)?;

    let probabilities = z.sigmoid()?;

    let shape = probabilities.get_shape().clone();
    let predictions_data = probabilities.get_data();

    let predictions_data = predictions_data
        .iter()
        .map(|i| if i.f64() >= 0.5 { D::one() } else { D::zero() })
        .collect();

    T::new(shape, predictions_data)
}
