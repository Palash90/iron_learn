use crate::commons::add_bias_term;
use crate::gradient_descent::gradient_descent;
use crate::numeric::FloatingPoint;
use crate::tensor::math::TensorMath;
use crate::Tensor;
use std::io;
use std::io::Write;

/// Predict outputs for `x` using linear model weights `w`.
pub fn predict_linear<T, D>(x: &T, w: &T) -> Result<T, String>
where
    T: Tensor<D> + TensorMath<D, MathOutput = T>,
    D: FloatingPoint,
{
    x.matmul(w)
}

/// Train a linear regression model using gradient descent.
///
/// - `x`, `y`: training data
/// - `w`: initial weights (consumed)
/// - `l`: learning rate
/// - `e`: number of epochs
pub fn linear_regression<T, D>(x: &T, y: &T, w: T, l: D, e: u32) -> Result<T, String>
where
    T: Tensor<D> + TensorMath<D, MathOutput = T>,
    D: FloatingPoint,
{
    let x_with_bias = add_bias_term(x)?;
    let mut weight = w;
    for i in 0..(e - 1) {
        print!("\rProcessing epoch: {i}/{e}");
        io::stdout().flush().unwrap();

        weight = gradient_descent(&x_with_bias, y, &weight, l, false)?;
    }

    Ok(gradient_descent(&x_with_bias, y, &weight, l, false)?)
}
