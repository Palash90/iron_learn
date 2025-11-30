use crate::tensor::Tensor;


pub fn denormalize_features(
    normalized_data: &Tensor<f64>,
    mean: &Vec<f64>,
    std: &Vec<f64>,
) -> Tensor<f64> {
    let shape = normalized_data.get_shape();
    let m = shape[0] as usize;
    let n = shape[1] as usize;
    let mut denormalized_data = vec![0.0; m * n];

    for j in 0..n {
        let mean = mean[j];
        let std_dev = std[j];

        for i in 0..m {
            let value = normalized_data.get_data()[i * n + j];
            denormalized_data[i * n + j] = value * std_dev + mean;
        }
    }

    Tensor::new(shape.clone(), denormalized_data).unwrap()
}

pub fn normalize_features(
    data: &Tensor<f64>,
    mean: &Vec<f64>,
    std: &Vec<f64>,
) -> Tensor<f64> {
    let shape = data.get_shape();
    let m = shape[0] as usize;
    let n = shape[1] as usize;
    let mut normalized_data = vec![0.0; m * n];

    let mut data_mean = vec![0.0; n];
    let mut data_std_dev = vec![0.0; n];

    for j in 0..n {
        let mean = mean[j];
        let std_dev = std[j];

        for i in 0..m {
            let value = data.get_data()[i * n + j];
            normalized_data[i * n + j] = if std_dev != 0.0 {
                (value - mean) / std_dev
            } else {
                0.0
            };
        }
    }

    Tensor::new(shape.clone(), normalized_data).unwrap()
}

pub fn normalize_features_mean_std(data: &Tensor<f64>) -> (Tensor<f64>, Vec<f64>, Vec<f64>) {
    let shape = data.get_shape();
    let m = shape[0] as usize;
    let n = shape[1] as usize;

    let mut data_mean = vec![0.0; n];
    let mut data_std_dev = vec![0.0; n];

    // For each feature
    for j in 0..n {
        let mut mean = 0.0;
        let mut variance = 0.0;

        // Calculate mean
        for i in 0..m {
            mean += data.get_data()[i * n + j];
        }
        mean /= m as f64;

        // Calculate variance
        for i in 0..m {
            let diff = data.get_data()[i * n + j] - mean;
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
