use crate::neural_network::ActivationFn;
use crate::neural_network::CoreNeuralNetType;
use crate::tensor::math::TensorMath;
use crate::Tensor;

pub fn sigmoid<T>(input: &T) -> Result<T, String>
where
    T: TensorMath<CoreNeuralNetType, MathOutput = T>,
{
    input.sigmoid()
}

pub fn sigmoid_prime<T>(output: &T) -> Result<T, String>
where
    T: TensorMath<CoreNeuralNetType, MathOutput = T> + Tensor<CoreNeuralNetType>,
{
    let one_minus_out = T::ones(&output.get_shape()).sub(output)?;
    output.multiply(&one_minus_out)
}

pub fn tanh<T>(input: &T) -> Result<T, String>
where
    T: TensorMath<CoreNeuralNetType, MathOutput = T>,
{
    input.tanh()
}

pub fn tanh_prime<T>(output: &T) -> Result<T, String>
where
    T: TensorMath<CoreNeuralNetType, MathOutput = T> + Tensor<CoreNeuralNetType>,
{
    let out_squared = output.multiply(output)?;

    let ones = T::ones(&output.get_shape());

    ones.sub(&out_squared)
}
