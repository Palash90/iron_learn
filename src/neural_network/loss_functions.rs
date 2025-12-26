use crate::neural_network::NeuralNetDataType;
use crate::tensor::math::TensorMath;
use crate::Numeric;
use crate::SignedNumeric;
use crate::Tensor;

pub trait LossFunction<D, T>
where
    D: Numeric,
    T: Tensor<D>,
{
    /// Calculates the loss value (used for reporting).
    fn loss(&self, actual: &T, predicted: &T) -> Result<T, String>;

    /// Calculates the derivative of the loss w.r.t the predicted output (used for backpropagation).
    fn loss_prime(&self, actual: &T, predicted: &T) -> Result<T, String>;
}

pub struct MeanSquaredErrorLoss;

impl<T: Tensor<D> + 'static, D> LossFunction<D, T> for MeanSquaredErrorLoss
where
    D: Numeric,
{
    fn loss(&self, actual: &T, predicted: &T) -> Result<T, String> {
        let error_diff = predicted.sub(actual).unwrap();
        let sq_err = error_diff.multiply(&error_diff).unwrap();

        let length = sq_err.get_shape().iter().product();

        sq_err.sum().unwrap().scale(D::one() / D::from_u32(length))
    }

    fn loss_prime(&self, actual: &T, predicted: &T) -> Result<T, String> {
        let n = actual.get_shape().iter().product();
        let factor = D::from_u32(2) / D::from_u32(n);

        predicted.sub(actual)?.scale(factor)
    }
}

pub struct BinaryCrossEntropy;

impl<T: Tensor<D> + 'static + TensorMath<D, MathOutput = T>, D> LossFunction<D, T>
    for BinaryCrossEntropy
where
    D: SignedNumeric + From<NeuralNetDataType>,
{
    fn loss(&self, y_true: &T, y_pred: &T) -> Result<T, String> {
        let shape = y_true.get_shape();
        let ones = T::ones(shape);

        // 1. Safety: Clip predictions to prevent ln(0) or ln(1-1)
        let epsilon = D::from(1e-7 as NeuralNetDataType);
        let one_minus_epsilon = D::from(1.0 - 1e-7 as NeuralNetDataType);
        let clipped_pred = y_pred.clip(epsilon, one_minus_epsilon)?;

        // 2. Term 1: y * ln(y_hat)
        let ln_pred = clipped_pred.ln()?;
        let term1 = y_true.multiply(&ln_pred)?;

        // 3. Term 2: (1 - y) * ln(1 - y_hat)
        let one_minus_y = ones.sub(y_true)?;
        let one_minus_pred = ones.sub(&clipped_pred)?;
        let ln_one_minus_pred = one_minus_pred.ln()?;
        let term2 = one_minus_y.multiply(&ln_one_minus_pred)?;

        // 4. Combine: -mean(term1 + term2)
        let combined = term1.add(&term2)?;
        let negative_one = -D::one();

        // Scale by -1 and reduce to scalar loss
        combined.scale(negative_one)?.sum()
    }

    fn loss_prime(&self, y_true: &T, y_pred: &T) -> Result<T, String> {
        let shape = y_true.get_shape();
        let ones = T::ones(shape);
        let epsilon = D::from(1e-7 as NeuralNetDataType);

        // Formula: (y_pred - y_true) / [y_pred * (1 - y_pred)]

        // Numerator: (y_pred - y_true)
        let numerator = y_pred.sub(y_true)?;

        // Denominator: y_pred * (1 - y_pred)
        let one_minus_pred = ones.sub(y_pred)?;
        let denominator = y_pred.multiply(&one_minus_pred)?;

        // Stability: Ensure we don't divide by zero
        let safe_denominator = denominator.clip(epsilon, D::one())?;

        // Note: This requires a 'div' method in your Tensor trait
        numerator.div(&safe_denominator)
    }
}
