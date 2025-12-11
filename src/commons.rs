use crate::tensor::Tensor;
use crate::Numeric;

pub fn denormalize_features<T: Tensor<f64>>(
    normalized_data: &T,
    mean: &Vec<f64>,
    std: &Vec<f64>,
) -> T {
    let shape = normalized_data.get_shape();
    let m = shape[0] as usize;
    let n = shape[1] as usize;
    let mut denormalized_data = vec![0.0; m * n];
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

pub fn normalize_features<T: Tensor<f64>>(data: &T, mean: &Vec<f64>, std: &Vec<f64>) -> T {
    let shape = data.get_shape();
    let m = shape[0] as usize;
    let n = shape[1] as usize;
    let mut normalized_data = vec![0.0; m * n];

    let mut data_mean = vec![0.0; n];
    let mut data_std_dev = vec![0.0; n];

    let data = data.get_data();

    for j in 0..n {
        let mean = mean[j];
        let std_dev = std[j];

        for i in 0..m {
            let value = data[i * n + j];
            normalized_data[i * n + j] = if std_dev != 0.0 {
                (value - mean) / std_dev
            } else {
                0.0
            };
        }
    }

    T::new(shape.clone(), normalized_data).unwrap()
}

pub fn normalize_features_mean_std<T: Tensor<f64>>(data: &T) -> (T, Vec<f64>, Vec<f64>) {
    let shape = data.get_shape();
    let m = shape[0] as usize;
    let n = shape[1] as usize;

    let mut data_mean = vec![0.0; n];
    let mut data_std_dev = vec![0.0; n];

    // For each feature
    for j in 0..n {
        let mut mean = 0.0;
        let mut variance = 0.0;
        let data = data.get_data();

        // Calculate mean
        for i in 0..m {
            mean += data[i * n + j];
        }
        mean /= m as f64;

        // Calculate variance
        for i in 0..m {
            let diff = data[i * n + j] - mean;
            variance += diff * diff;
        }
        variance /= m as f64;
        let std_dev = variance.sqrt();

        data_mean[j] = mean;
        data_std_dev[j] = std_dev;
    }

    let normalized_data = normalize_features(data, &data_mean, &data_std_dev);

    (normalized_data, data_mean, data_std_dev)
}

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