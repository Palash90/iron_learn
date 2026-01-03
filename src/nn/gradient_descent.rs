use crate::numeric::FloatingPoint;
use crate::tensor::math::TensorMath;
use crate::tensor::Tensor;

/// Perform a single gradient descent update step.
///
/// - `x`: input features (may include bias column if already added)
/// - `y`: target outputs
/// - `w`: current weights
/// - `l`: learning rate
/// - `logistic`: whether to use a logistic (sigmoid) activation
pub fn gradient_descent<D, T>(x: &T, y: &T, w: &T, l: D, logistic: bool) -> Result<T, String>
where
    D: FloatingPoint,
    T: Tensor<D> + TensorMath<D, MathOutput = T>,
{
    let data_size = *(x.get_shape().first().ok_or("X must have a shape").unwrap());

    let lines = x.mul(w)?;

    let prediction = match logistic {
        true => lines.sigmoid(),
        false => Ok(lines),
    }?;

    let loss = prediction.sub(y)?;

    let gradient_raw = x.t()?.mul(&loss)?;

    let d = gradient_raw.scale(l / D::from_u32(data_size))?;

    w.sub(&d)
}
