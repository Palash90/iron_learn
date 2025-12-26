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

        predicted.sub(actual).unwrap().scale(factor)
    }
}

pub struct BinaryCrossEntropy;

impl<T: Tensor<D> + 'static + TensorMath<D, MathOutput = T>, D> LossFunction<D, T>
    for BinaryCrossEntropy
where
    D: SignedNumeric + From<NeuralNetDataType>,
{
    fn loss(&self, y_true: &T, y_pred: &T) -> Result<T, String> {
        // println!();
        //  println!("Loss Function");
        //  println!("Actual");
        //   y_true.print_matrix();

        //    println!("Predicted");
        //   y_pred.print_matrix();

        let shape = y_true.get_shape();
        let ones = T::ones(shape);

        // 1. Safety: Clip predictions to prevent ln(0) or ln(1-1)
        let epsilon = D::from(1e-7 as NeuralNetDataType);
        let one_minus_epsilon = D::from(1.0 - 1e-7 as NeuralNetDataType);
        let clipped_pred = y_pred.clip(epsilon, one_minus_epsilon).unwrap();

        //    println!("Predicted clipped");
        //   clipped_pred.print_matrix();

        // 2. Term 1: y * ln(y_hat)
        let ln_pred = clipped_pred.ln().unwrap();

        //   println!("Predicted ln");
        //   clipped_pred.print_matrix();

        let term1 = y_true.multiply(&ln_pred).unwrap();

        //    println!("term1");
        //   term1.print_matrix();

        // 3. Term 2: (1 - y) * ln(1 - y_hat)
        let one_minus_y = ones.sub(y_true).unwrap();

        //    println!("one_minus_y");
        //    one_minus_y.print_matrix();

        let one_minus_pred = ones.sub(&clipped_pred).unwrap();

        //    println!("one_minus_pred");
        //    one_minus_pred.print_matrix();

        let ln_one_minus_pred = one_minus_pred.ln().unwrap();

        //    println!("ln_one_minus_pred");
        //    ln_one_minus_pred.print_matrix();

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
        let epsilon = D::from(1e-12 as NeuralNetDataType);
        let one = D::from(1.0 as NeuralNetDataType);
        let one_minus_epsilon = D::from((1.0 - 1e-12) as NeuralNetDataType);

        // 2. Stability: Clip y_pred just like the Python version
        // This prevents division by zero in the denominator later
        let clipped_pred = y_pred.clip(epsilon, one_minus_epsilon).unwrap();

        // 3. Calculate: (clipped_pred - y_true) / [clipped_pred * (1 - clipped_pred)]
        let numerator = clipped_pred.sub(y_true).unwrap();
        let one_minus_pred = ones.sub(&clipped_pred).unwrap();
        let denominator = clipped_pred.multiply(&one_minus_pred).unwrap();

        let mut result = numerator.div(&denominator).unwrap();

        // 4. Normalization: Divide by y.size to get the mean gradient
        let size = D::from(shape.iter().product::<u32>() as NeuralNetDataType);
        result = result.scale(D::one() / size).unwrap();

        Ok(result)
    }
}
