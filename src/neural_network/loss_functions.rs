use crate::Numeric;
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
        let error_diff = actual.sub(predicted).unwrap();
        let sq_err = error_diff.multiply(&error_diff).unwrap();

        let length = sq_err.get_shape()[1];

        sq_err.sum().unwrap().scale(D::one() / D::from_u32(length))
    }

    fn loss_prime(&self, actual: &T, predicted: &T) -> Result<T, String> {
        predicted
            .sub(actual)
            .unwrap()
            .scale(D::from_u32(2) / D::from_u32(actual.get_shape()[0]))
    }
}
