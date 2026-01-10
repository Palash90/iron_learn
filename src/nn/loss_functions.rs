use serde::{Deserialize, Serialize};

use crate::numeric::FloatingPoint;
use crate::tensor::math::TensorMath;
use crate::LossFn;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum LossFunctionType {
    MeanSquaredError,
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
}

pub fn get_loss_function<T, D>(loss_type: &LossFunctionType) -> (LossFn<T>, LossFn<T>)
where
    T: TensorMath<D, MathOutput = T>,
    D: FloatingPoint,
{
    match loss_type {
        LossFunctionType::MeanSquaredError => (mse, mse_prime),
        LossFunctionType::BinaryCrossEntropy => (bce, bce_prime),
        LossFunctionType::CategoricalCrossEntropy => (cce, cce_prime),
    }
}

pub fn mse<T, D>(actual: &T, predicted: &T) -> Result<T, String>
where
    T: TensorMath<D, MathOutput = T>,
    D: FloatingPoint,
{
    let error_diff = predicted.sub(actual).unwrap();
    let sq_err = error_diff.mul(&error_diff).unwrap();

    let length = sq_err.get_shape().iter().product();

    sq_err.sum().unwrap().scale(D::one() / D::from_u32(length))
}

pub fn mse_prime<T, D>(actual: &T, predicted: &T) -> Result<T, String>
where
    T: TensorMath<D, MathOutput = T>,
    D: FloatingPoint,
{
    let n = actual.get_shape().iter().product();
    let factor = D::from_u32(2) / D::from_u32(n);

    predicted.sub(actual).unwrap().scale(factor)
}

pub fn bce<T, D>(y_true: &T, y_pred: &T) -> Result<T, String>
where
    T: TensorMath<D, MathOutput = T>,
    D: FloatingPoint,
{
    let shape = y_true.get_shape();
    let ones = T::ones(shape);

    let epsilon = D::from_f64(1e-7);
    let one_minus_epsilon = D::from_f64(1.0 - 1e-7);
    let clipped_pred = y_pred.clip(epsilon, one_minus_epsilon).unwrap();

    let ln_pred = clipped_pred.ln().unwrap();

    let term1 = y_true.mul(&ln_pred).unwrap();

    let one_minus_y = ones.sub(y_true).unwrap();

    let one_minus_pred = ones.sub(&clipped_pred).unwrap();

    let ln_one_minus_pred = one_minus_pred.ln().unwrap();

    let term2 = one_minus_y.mul(&ln_one_minus_pred).unwrap();

    let combined = term1.add(&term2).unwrap();
    let negative_one = -D::one();

    let length = combined.get_shape().iter().product();

    combined
        .scale(negative_one)
        .unwrap()
        .sum()
        .unwrap()
        .scale(D::one() / D::from_u32(length))
}

pub fn bce_prime<T, D>(y_true: &T, y_pred: &T) -> Result<T, String>
where
    T: TensorMath<D, MathOutput = T>,
    D: FloatingPoint,
{
    let shape = y_true.get_shape();
    let ones = T::ones(shape);

    let epsilon = D::from_f64(1e-12);
    let one_minus_epsilon = D::from_f64(1.0 - 1e-12);

    let clipped_pred = y_pred.clip(epsilon, one_minus_epsilon).unwrap();

    let numerator = clipped_pred.sub(y_true).unwrap();
    let one_minus_pred = ones.sub(&clipped_pred).unwrap();
    let denominator = clipped_pred.mul(&one_minus_pred).unwrap();

    let mut result = numerator.div(&denominator).unwrap();

    let size = D::from_u32(shape.iter().product::<u32>());
    result = result.scale(D::one() / size).unwrap();

    Ok(result)
}

pub fn cce<T, D>(y_true: &T, y_pred: &T) -> Result<T, String>
where
    T: TensorMath<D, MathOutput = T>,
    D: FloatingPoint,
{
    let epsilon = D::from_f64(1e-12);
    let clipped_pred = y_pred.clip(epsilon, D::one()).unwrap();

    let ln_pred = clipped_pred.ln().unwrap();
    let product = y_true.mul(&ln_pred).unwrap();

    let length = product.get_shape().iter().product();

    let negative_one = -D::one();
    product
        .scale(negative_one)
        .unwrap()
        .sum()
        .unwrap()
        .scale(D::one() / D::from_u32(length))
}

pub fn cce_prime<T, D>(y_true: &T, y_pred: &T) -> Result<T, String>
where
    T: TensorMath<D, MathOutput = T>,
    D: FloatingPoint,
{
    let shape = y_true.get_shape();
    let mut result = y_pred.sub(y_true).unwrap();

    let batch_size = D::from_u32(shape[0]);
    result = result.scale(D::one() / batch_size).unwrap();

    Ok(result)
}
