use crate::numeric::FloatingPoint;
use crate::tensor::Tensor;
use crate::Numeric;

/// Denormalize features using per-feature `mean` and `std` vectors.
///
/// Returns a new tensor where each feature column is scaled back to the
/// original value range by computing `value * std[j] + mean[j]`.
pub fn denormalize_features<T, D>(normalized_data: &T, mean: &[D], std: &[D]) -> T
where
    T: Tensor<D>,
    D: FloatingPoint,
{
    let shape = normalized_data.get_shape();
    let m = shape[0] as usize;
    let n = shape[1] as usize;
    let mut denormalized_data = vec![D::zero(); m * n];
    let data = normalized_data.get_data();

    for j in 0..n {
        let mean = mean[j];
        let std_dev = std[j];

        for i in 0..m {
            let value = data[i * n + j];
            denormalized_data[i * n + j] = value * std_dev + mean;
        }
    }

    T::new(shape.clone(), denormalized_data).unwrap()
}

/// Normalize features using provided `mean` and `std` per feature.
///
/// For each element returns `(value - mean[j]) / std[j]` when `std[j] != 0`,
/// otherwise `0.0` to avoid division-by-zero.
pub fn normalize_features<T, D>(data: &T, mean: &[D], std: &[D]) -> T
where
    T: Tensor<D>,
    D: FloatingPoint,
{
    let shape = data.get_shape();
    let m = shape[0] as usize;
    let n = shape[1] as usize;
    let mut normalized_data = vec![D::zero(); m * n];

    let data = data.get_data();

    for j in 0..n {
        let mean = mean[j];
        let std_dev = std[j];

        for i in 0..m {
            let value = data[i * n + j];
            normalized_data[i * n + j] = if std_dev != D::zero() {
                (value - mean) / std_dev
            } else {
                D::zero()
            };
        }
    }

    T::new(shape.clone(), normalized_data).unwrap()
}

/// Compute per-feature mean and standard deviation, and return the
/// normalized tensor along with the `mean` and `std` vectors.
pub fn normalize_features_mean_std<T, D>(data: &T) -> (T, Vec<D>, Vec<D>)
where
    T: Tensor<D>,
    D: FloatingPoint,
{
    let shape = data.get_shape();
    let m = shape[0] as usize;
    let n = shape[1] as usize;

    let mut data_mean = vec![D::zero(); n];
    let mut data_std_dev = vec![D::zero(); n];

    let data_vec = data.get_data();

    // For each feature
    for j in 0..n {
        let mut mean = D::zero();
        let mut variance = D::zero();

        // Calculate mean
        for i in 0..m {
            mean = mean + data_vec[i * n + j];
        }
        mean = mean / D::from_u32(m as u32);

        // Calculate variance
        for i in 0..m {
            let diff = data_vec[i * n + j] - mean;
            variance = variance + diff * diff;
        }
        variance = variance / D::from_u32(m as u32);
        let std_dev = variance.sqrt();

        data_mean[j] = mean;
        data_std_dev[j] = std_dev;
    }

    let normalized_data = normalize_features(data, &data_mean, &data_std_dev);

    (normalized_data, data_mean, data_std_dev)
}

/// Add a bias column of ones to the input tensor `x` and return the new
/// tensor. The new tensor will have shape `[m, n+1]` where the first column
/// contains ones.
pub fn add_bias_term<T, U>(x: &T) -> Result<T, String>
where
    T: Tensor<U>,
    U: Numeric,
{
    let shape = x.get_shape();
    let m = shape[0] as usize;
    let n = shape[1] as usize;
    let x_data = x.get_data();
    let mut data = Vec::<U>::with_capacity(m * (n + 1));

    for i in 0..m {
        data.push(U::one());
        for j in 0..n {
            data.push(x_data[i * n + j]);
        }
    }

    T::new(vec![shape[0], shape[1] + 1], data)
}
