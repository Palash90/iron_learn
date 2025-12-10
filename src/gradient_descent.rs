use crate::tensor::Tensor;

pub fn add_bias_term<T: Tensor<f64>>(x: &T) -> Result<T, String> {
    let shape = x.get_shape();
    let m = shape[0] as usize;
    let n = shape[1] as usize;
    let x_data = x.get_data();
    let mut data = Vec::with_capacity(m * (n + 1));

    for i in 0..m {
        data.push(1.0);
        for j in 0..n {
            data.push(x_data[i * n + j]);
        }
    }

    T::new(vec![shape[0], shape[1] + 1], data)
}

pub fn gradient_descent<T: Tensor<f64>>(
    x: &T,
    y: &T,
    w: &T,
    l: f64,
    logistic: bool,
) -> Result<T, String> {
    let data_size = *(x.get_shape().first().ok_or("X must have a shape")?) as f64;

    let lines = x.mul(w)?;

    let prediction = match logistic {
        true => lines.sigmoid(),
        false => Ok(lines),
    }?;

    let loss = prediction.sub(y)?;

    let gradient_raw = x.t()?.mul(&loss)?;

    let d = gradient_raw.scale(l / data_size)?;

    w.sub(&d)
}

pub fn linear_regression<T: Tensor<f64>>(x: &T, y: &T, w: T, l: f64, e: u32) -> Result<T, String> {
    let x_with_bias = add_bias_term(x)?;
    let mut weight = w;
    for _ in 0..(e - 1) {
        weight = gradient_descent(&x_with_bias, y, &weight, l, false).unwrap();
    }

    Ok(gradient_descent(&x_with_bias, y, &weight, l, false).unwrap())
}

pub fn logistic_regression<T: Tensor<f64>>(
    x: &T,
    y: &T,
    w: T,
    l: f64,
    e: u32,
) -> Result<T, String> {
    let x_with_bias = add_bias_term(x)?;
    let mut weight = w;

    for _ in 0..(e - 1) {
        weight = gradient_descent(&x_with_bias, y, &weight, l, true).unwrap();
    }

    Ok(gradient_descent(&x_with_bias, y, &weight, l, true).unwrap())
}

pub fn predict_linear<T: Tensor<f64>>(x: &T, w: &T) -> Result<T, String> {
    let x_with_bias = add_bias_term(x)?;
    x_with_bias.mul(w)
}

pub fn predict_logistic<T: Tensor<f64>>(x: &T, w: &T) -> Result<T, String> {
    let x_with_bias = add_bias_term(x)?;

    let z = x_with_bias.mul(w)?;

    let probabilities = z.sigmoid()?;

    let shape = probabilities.get_shape().clone();
    let predictions_data = probabilities
        .get_data()
        .iter()
        .map(|&p| if p >= 0.5 { 1.0 } else { 0.0 })
        .collect();

    T::new(shape, predictions_data)
}
