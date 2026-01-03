use crate::numeric::FloatingPoint;
use crate::tensor::math::TensorMath;
use crate::Numeric;
use crate::Tensor;

/// Trait describing a loss function used for training and backpropagation.
///
/// Implementors must provide methods to compute the scalar loss tensor and
/// the derivative (loss prime) used as the starting point for backprop.
pub trait LossFunction<T, D>
where
    D: Numeric,
    T: Tensor<D>,
{
    /// Calculates the loss value (used for reporting).
    fn loss(&self, actual: &T, predicted: &T) -> Result<T, String>;

    /// Calculates the derivative of the loss w.r.t the predicted output (used for backpropagation).
    fn loss_prime(&self, actual: &T, predicted: &T) -> Result<T, String>;
}

/// Mean squared error loss implementation.
pub struct MeanSquaredErrorLoss;

impl<T: Tensor<D> + 'static, D> LossFunction<T, D> for MeanSquaredErrorLoss
where
    D: Numeric,
    T: Tensor<D>,
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

        predicted.sub(actual).unwrap().scale(factor)
    }
}

/// Binary cross-entropy loss implementation (stable variant with clipping).
pub struct BinaryCrossEntropy;

impl<T: Tensor<D> + 'static + TensorMath<D, MathOutput = T>, D> LossFunction<T, D>
    for BinaryCrossEntropy
where
    D: FloatingPoint,
    T: Tensor<D>,
{
    fn loss(&self, y_true: &T, y_pred: &T) -> Result<T, String> {
        let shape = y_true.get_shape();
        let ones = T::ones(shape);

        // 1. Safety: Clip predictions to prevent ln(0) or ln(1-1)
        let epsilon = D::from_f64(1e-7);
        let one_minus_epsilon = D::from_f64(1.0 - 1e-7);
        let clipped_pred = y_pred.clip(epsilon, one_minus_epsilon).unwrap();

        // 2. Term 1: y * ln(y_hat)
        let ln_pred = clipped_pred.ln().unwrap();

        let term1 = y_true.multiply(&ln_pred).unwrap();

        // 3. Term 2: (1 - y) * ln(1 - y_hat)
        let one_minus_y = ones.sub(y_true).unwrap();

        let one_minus_pred = ones.sub(&clipped_pred).unwrap();

        let ln_one_minus_pred = one_minus_pred.ln().unwrap();

        let term2 = one_minus_y.multiply(&ln_one_minus_pred).unwrap();

        // 4. Combine: -mean(term1 + term2)
        let combined = term1.add(&term2).unwrap();
        let negative_one = -D::one();

        // Scale by -1 and reduce to scalar loss
        combined.scale(negative_one).unwrap().sum()
    }

    fn loss_prime(&self, y_true: &T, y_pred: &T) -> Result<T, String> {
        let shape = y_true.get_shape();
        let ones = T::ones(shape);

        // 1. Setup constants
        let epsilon = D::from_f64(1e-12);
        let one_minus_epsilon = D::from_f64(1.0 - 1e-12);

        // 2. Stability: Clip y_pred just like the Python version
        // This prevents division by zero in the denominator later
        let clipped_pred = y_pred.clip(epsilon, one_minus_epsilon).unwrap();

        // 3. Calculate: (clipped_pred - y_true) / [clipped_pred * (1 - clipped_pred)]
        let numerator = clipped_pred.sub(y_true).unwrap();
        let one_minus_pred = ones.sub(&clipped_pred).unwrap();
        let denominator = clipped_pred.multiply(&one_minus_pred).unwrap();

        let mut result = numerator.div(&denominator).unwrap();

        // 4. Normalization: Divide by y.size to get the mean gradient
        let size = D::from_u32(shape.iter().product::<u32>());
        result = result.scale(D::one() / size).unwrap();

        Ok(result)
    }
}
