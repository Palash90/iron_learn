use crate::neural_network::NeuralNetDataType;
use crate::tensor::math::TensorMath;
use crate::Tensor;

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
    let res = output.multiply(&one_minus_out).unwrap();

    let epsilon = T::new(output.get_shape().clone(), vec![1e-10; output.get_shape().iter().product::<u32>() as usize]).unwrap();
    res.add(&epsilon)
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
