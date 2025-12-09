use crate::tensor_commons::TensorOps; // Assuming TensorOps is visible here

// --- Helper Functions ---

pub fn normalize_features<T: TensorOps<f64>>(x: &T) -> Result<T, String> {
    let shape = x.get_shape();
    let m = shape[0] as usize; // number of examples
    let n = shape[1] as usize; // number of features
    let x_data = x.get_data(); // Use the assumed get_data()
    let mut normalized_data = vec![0.0; m * n];

    // For each feature
    for j in 0..n {
        let mut mean = 0.0;
        let mut variance = 0.0;

        // Calculate mean
        for i in 0..m {
            mean += x_data[i * n + j];
        }
        mean /= m as f64;

        // Calculate variance
        for i in 0..m {
            let diff = x_data[i * n + j] - mean;
            variance += diff * diff;
        }
        variance /= m as f64;
        let std_dev = variance.sqrt();

        // Normalize feature
        for i in 0..m {
            normalized_data[i * n + j] = if std_dev > 1e-8 {
                (x_data[i * n + j] - mean) / std_dev
            } else {
                x_data[i * n + j] - mean
            };
        }
    }

    // Use the assumed new()
    T::new(shape.clone(), normalized_data)
}

pub fn add_bias_term<T: TensorOps<f64>>(x: &T) -> Result<T, String> {
    let shape = x.get_shape();
    let m = shape[0] as usize; // number of examples
    let n = shape[1] as usize; // number of features
    let x_data = x.get_data(); // Use the assumed get_data()
    let mut data = Vec::with_capacity(m * (n + 1));

    // For each example
    for i in 0..m {
        // Add 1 as the bias term
        data.push(1.0);
        // Add the original features
        for j in 0..n {
            data.push(x_data[i * n + j]);
        }
    }

    // Use the assumed new()
    T::new(vec![shape[0], shape[1] + 1], data)
}

// --- Core Training Function ---

pub fn gradient_descent<T: TensorOps<f64>>(
    x: &T,
    y: &T,
    w: &T,
    l: f64,
    logistic: bool,
) -> Result<T, String> {
    // We assume get_shape() returns a non-empty vector
    let data_size = *(x.get_shape().first().ok_or("X must have a shape")?) as f64;
    
    // Calculate z = X * w
    let lines = x.mul(w)?;

    // Calculate prediction: h(z) = z (linear) or h(z) = sigmoid(z) (logistic)
    let prediction = match logistic {
        true => lines.sigmoid(),
        false => Ok(lines),
    }?;

    // Calculate loss/error: (h(z) - y)
    let loss = prediction.sub(y)?;
    
    // Calculate the gradient d = X.T * loss
    let gradient_raw = x.t()?.mul(&loss)?;

    // Scale gradient: (1/m) * (X.T * loss)
    let d = gradient_raw.scale(l / data_size)?; // Scale by learning rate (l) / data size (m)

    // Update weights: w = w - d
    w.sub(&d)
}

// --- Regression API Functions ---

pub fn linear_regression<T: TensorOps<f64>>(x: &T, y: &T, w: &T, l: f64) -> Result<T, String> {
    let x_normalized = normalize_features(x)?;
    let x_with_bias = add_bias_term(&x_normalized)?;
    gradient_descent(&x_with_bias, y, w, l, false)
}

pub fn logistic_regression<T: TensorOps<f64>>(
    x: &T,
    y: &T,
    w: &T,
    l: f64,
) -> Result<T, String> {
    let x_normalized = normalize_features(x)?;
    let x_with_bias = add_bias_term(&x_normalized)?;
    gradient_descent(&x_with_bias, y, w, l, true)
}

// --- Prediction API Functions ---

pub fn predict_linear<T: TensorOps<f64>>(x: &T, w: &T) -> Result<T, String> {
    // First normalize features
    let x_normalized = normalize_features(x)?;
    // Then add bias term
    let x_with_bias = add_bias_term(&x_normalized)?;
    // Calculate prediction: X_bias * w
    x_with_bias.mul(w)
}

pub fn predict_logistic<T: TensorOps<f64>>(x: &T, w: &T) -> Result<T, String> {
    // First normalize features
    let x_normalized = normalize_features(x)?;
    // Then add bias term
    let x_with_bias = add_bias_term(&x_normalized)?;
    
    // Calculate z = X_bias * w
    let z = x_with_bias.mul(w)?;
    
    // Calculate probabilities: h(z) = sigmoid(z)
    let probabilities = z.sigmoid()?;

    // Convert probabilities to binary predictions (0 or 1)
    let shape = probabilities.get_shape().clone();
    let predictions_data = probabilities
        .get_data()
        .iter()
        .map(|&p| if p >= 0.5 { 1.0 } else { 0.0 })
        .collect();

    // Create a new Tensor with the binary predictions
    T::new(shape, predictions_data)
}