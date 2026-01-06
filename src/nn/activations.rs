use serde::{Deserialize, Serialize};

use crate::numeric::FloatingPoint;
use crate::tensor::math::TensorMath;
use crate::{ActivationFn, Tensor};

/// Enum describing available activation types used by `ActivationLayer`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayerType {
    Sigmoid,
    Tanh,
    Linear,
    Sin,
    ReLU,
    Softmax,
}
/// Return the activation function and its derivative for `layer`.
pub fn get_activations<T, D>(layer: &LayerType) -> (ActivationFn<T>, ActivationFn<T>)
where
    T: TensorMath<D, MathOutput = T> + Tensor<D>,
    D: FloatingPoint,
{
    match layer {
        LayerType::Sigmoid => (sigmoid, sigmoid_prime),
        LayerType::Tanh => (tanh, tanh_prime),
        LayerType::Sin => (sin, cos),
        LayerType::Linear => (
            |x: &T| Ok(T::zeroes(x.get_shape()).add(x).unwrap()),
            |x: &T| Ok(T::ones(x.get_shape())),
        ),
        LayerType::ReLU => (relu, relu_prime),
        LayerType::Softmax => (softmax, softmax_prime),
    }
}

/// Element-wise ReLU activation: f(x) = max(0, x)
pub fn relu<T, D>(input: &T) -> Result<T, String>
where
    T: TensorMath<D, MathOutput = T> + Tensor<D>,
    D: FloatingPoint,
{
    input.relu()
}

/// Derivative of ReLU; expects the original input (or the activation output)
/// f'(x) = 1 if x > 0, else 0
pub fn relu_prime<T, D>(input: &T) -> Result<T, String>
where
    T: TensorMath<D, MathOutput = T> + Tensor<D>,
    D: FloatingPoint,
{
    input.greater_than_zero_mask()
}

/// Element-wise sigmoid activation.
pub fn sigmoid<T, D>(input: &T) -> Result<T, String>
where
    T: TensorMath<D, MathOutput = T>,
    D: FloatingPoint,
{
    input.sigmoid()
}

/// Derivative of sigmoid; expects the activation output as input.
pub fn sigmoid_prime<T, D>(output: &T) -> Result<T, String>
where
    T: TensorMath<D, MathOutput = T> + Tensor<D>,
    D: FloatingPoint,
{
    let one_minus_out = T::ones(output.get_shape()).sub(output)?;
    output.mul(&one_minus_out)
}

/// Element-wise hyperbolic tangent activation.
pub fn tanh<T, D>(input: &T) -> Result<T, String>
where
    T: TensorMath<D, MathOutput = T>,
    D: FloatingPoint,
{
    input.tanh()
}

/// Derivative of `tanh`, expects activation output as input.
pub fn tanh_prime<T, D>(output: &T) -> Result<T, String>
where
    T: TensorMath<D, MathOutput = T> + Tensor<D>,
    D: FloatingPoint,
{
    let out_squared = output.mul(output)?;

    let ones = T::ones(output.get_shape());

    ones.sub(&out_squared)
}

/// Element-wise sine activation.
pub fn sin<T, D>(input: &T) -> Result<T, String>
where
    T: TensorMath<D, MathOutput = T>,
    D: FloatingPoint,
{
    input.sin()
}

/// Element-wise cosine function (used as derivative for `sin`).
pub fn cos<T, D>(output: &T) -> Result<T, String>
where
    T: TensorMath<D, MathOutput = T> + Tensor<D>,
    D: FloatingPoint,
{
    output.cos()
}

/// Aggregating activation: f(x_i) = exp(x_i) / sum(exp(x_j))
pub fn softmax<T, D>(input: &T) -> Result<T, String>
where
    T: TensorMath<D, MathOutput = T> + Tensor<D>,
    D: FloatingPoint,
{
    // 1. Get the raw data and shape
    let data = input.get_data();
    let shape = input.get_shape();
    let rows = shape[0] as usize;
    let cols = shape[1] as usize;

    let mut result_data = Vec::with_capacity(data.len());

    // 2. Process each row (sample) independently to handle batches
    for r in 0..rows {
        let row_slice = &data[r * cols..(r + 1) * cols];

        // Find max in row for numerical stability: exp(x - max)
        let max_val = row_slice
            .iter()
            .fold(D::neg_infinity(), |a, &b| if a > b { a } else { b });

        // Calculate exp and sum for the row
        let mut row_exp: Vec<D> = row_slice.iter().map(|&x| (x - max_val).exp()).collect();

        let row_sum: D = row_exp.iter().fold(D::zero(), |a, &b| a + b);

        // Normalize row
        for val in row_exp.iter_mut() {
            *val = *val / row_sum;
        }
        result_data.extend(row_exp);
    }

    T::new(shape.to_vec(), result_data)
}

/// Identity derivative for Softmax when paired with Cross-Entropy
pub fn softmax_prime<T, D>(output: &T) -> Result<T, String>
where
    T: TensorMath<D, MathOutput = T> + Tensor<D>,
    D: FloatingPoint,
{
    Ok(T::ones(output.get_shape()))
}
