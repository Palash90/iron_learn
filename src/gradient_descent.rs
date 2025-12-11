use crate::tensor::math::TensorMath;
use crate::tensor::Tensor;
use crate::Numeric;

use crate::commons::add_bias_term;

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

pub fn predict_linear<T>(x: &T, w: &T) -> Result<T, String>
where
    T: Tensor<f64> + TensorMath<f64, MathOutput = T>,
{
    x.mul(w)
}

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
