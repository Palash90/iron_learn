use serde::{Deserialize, Serialize};

use crate::neural_network::NeuralNetDataType;
use crate::tensor::math::TensorMath;
use crate::{ActivationFn, Tensor};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayerType {
    Sigmoid,
    Tanh,
    Linear,
    Sin,
}

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

pub fn sigmoid<T>(input: &T) -> Result<T, String>
where
    T: TensorMath<NeuralNetDataType, MathOutput = T>,
{
    input.sigmoid()
}

pub fn sigmoid_prime<T>(output: &T) -> Result<T, String>
where
    T: TensorMath<NeuralNetDataType, MathOutput = T> + Tensor<NeuralNetDataType>,
{
    let one_minus_out = T::ones(&output.get_shape()).sub(output)?;
    let res = output.multiply(&one_minus_out);

    res
}

pub fn tanh<T>(input: &T) -> Result<T, String>
where
    T: TensorMath<NeuralNetDataType, MathOutput = T>,
{
    input.tanh()
}

pub fn tanh_prime<T>(output: &T) -> Result<T, String>
where
    T: TensorMath<NeuralNetDataType, MathOutput = T> + Tensor<NeuralNetDataType>,
{
    let out_squared = output.multiply(output)?;

    let ones = T::ones(&output.get_shape());

    ones.sub(&out_squared)
}

pub fn sin<T>(input: &T) -> Result<T, String>
where
    T: TensorMath<NeuralNetDataType, MathOutput = T>,
{
    input.sin()
}

pub fn cos<T>(output: &T) -> Result<T, String>
where
    T: TensorMath<NeuralNetDataType, MathOutput = T> + Tensor<NeuralNetDataType>,
{
    output.cos()
}
