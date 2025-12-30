use serde::{Deserialize, Serialize};

use crate::neural_network::NeuralNetDataType;
use crate::tensor::math::TensorMath;
use crate::{ActivationFn, Tensor};

/// Enum describing available activation types used by `ActivationLayer`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayerType {
    Sigmoid,
    Tanh,
    Linear,
    Sin,
}
/// Return the activation function and its derivative for `layer`.
pub fn get_activations<T>(layer: &LayerType) -> (ActivationFn<T>, ActivationFn<T>)
where
    T: TensorMath<NeuralNetDataType, MathOutput = T> + Tensor<NeuralNetDataType>,
{
    match layer {
        LayerType::Sigmoid => (sigmoid, sigmoid_prime),
        LayerType::Tanh => (tanh, tanh_prime),
        LayerType::Sin => (sin, cos),
        LayerType::Linear => (
            |x: &T| Ok(T::zeroes(x.get_shape()).add(x).unwrap()),
            |x: &T| Ok(T::ones(x.get_shape())),
        ),
    }
}

/// Element-wise sigmoid activation.
pub fn sigmoid<T>(input: &T) -> Result<T, String>
where
    T: TensorMath<NeuralNetDataType, MathOutput = T>,
{
    input.sigmoid()
}

/// Derivative of sigmoid; expects the activation output as input.
pub fn sigmoid_prime<T>(output: &T) -> Result<T, String>
where
    T: TensorMath<NeuralNetDataType, MathOutput = T> + Tensor<NeuralNetDataType>,
{
    let one_minus_out = T::ones(&output.get_shape()).sub(output)?;
    let res = output.multiply(&one_minus_out);

    res
}

/// Element-wise hyperbolic tangent activation.
pub fn tanh<T>(input: &T) -> Result<T, String>
where
    T: TensorMath<NeuralNetDataType, MathOutput = T>,
{
    input.tanh()
}

/// Derivative of `tanh`, expects activation output as input.
pub fn tanh_prime<T>(output: &T) -> Result<T, String>
where
    T: TensorMath<NeuralNetDataType, MathOutput = T> + Tensor<NeuralNetDataType>,
{
    let out_squared = output.multiply(output)?;

    let ones = T::ones(&output.get_shape());

    ones.sub(&out_squared)
}

/// Element-wise sine activation.
pub fn sin<T>(input: &T) -> Result<T, String>
where
    T: TensorMath<NeuralNetDataType, MathOutput = T>,
{
    input.sin()
}

/// Element-wise cosine function (used as derivative for `sin`).
pub fn cos<T>(output: &T) -> Result<T, String>
where
    T: TensorMath<NeuralNetDataType, MathOutput = T> + Tensor<NeuralNetDataType>,
{
    output.cos()
}
