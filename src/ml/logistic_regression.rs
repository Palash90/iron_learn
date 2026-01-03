use std::time::Instant;

use crate::{
    commons::add_bias_term,
    examples::{
        contexts::GLOBAL_CONTEXT, read_file::deserialize_data_double_precision,
        types::DataDoublePrecision,
    },
    gradient_descent::gradient_descent,
    tensor::math::TensorMath,
    Tensor,
};

/// Train a logistic regression model using gradient descent.
///
/// Same parameters as `linear_regression`, but applies a sigmoid
/// activation during training.
pub fn logistic_regression<T>(x: &T, y: &T, w: T, l: f64, e: u32) -> Result<T, String>
where
    T: Tensor<f64> + TensorMath<f64, MathOutput = T>,
{
    let x_with_bias = add_bias_term(x)?;
    let mut weight = w;

    for _ in 0..(e - 1) {
        weight = gradient_descent(&x_with_bias, y, &weight, l, true).unwrap();
    }

    Ok(gradient_descent(&x_with_bias, y, &weight, l, true).unwrap())
}

/// Predict binary labels for `x` using logistic model weights `w`.
///
/// Returns a tensor of predicted labels (0.0 or 1.0).
pub fn predict_logistic<T>(x: &T, w: &T) -> Result<T, String>
where
    T: Tensor<f64> + TensorMath<f64, MathOutput = T>,
{
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

/// Run logistic regression using configuration from the global context.
///
/// The function reads data and runtime settings from `GLOBAL_CONTEXT`,
/// trains a logistic model, evaluates on test data (if present), and
/// prints summary metrics. Returns `Ok(())` on success or an error string.
pub fn run_logistic<T>() -> Result<(), String>
where
    T: Tensor<f64> + TensorMath<f64, MathOutput = T>,
{
    let l = GLOBAL_CONTEXT
        .get()
        .ok_or("GLOBAL_CONTEXT not initialized")?
        .learning_rate;
    let e = GLOBAL_CONTEXT.get().unwrap().epochs;
    let data_path = &GLOBAL_CONTEXT.get().unwrap().data_path;

    let xy: DataDoublePrecision = deserialize_data_double_precision(&data_path)
        .map_err(|e| format!("Data deserialization error: {}", e))?;

    print!("\nLogistic Regression\n");
    print!("Number of examples (m): {}\n", xy.m);
    print!("Number of features (n): {}\n", xy.n);
    print!("Total X values length: {}\n", xy.x.len());
    print!("Total Y values length: {}\n", xy.y.len());

    let x = T::new(vec![xy.m, xy.n], xy.x.clone())?;
    let y = T::new(vec![xy.m, 1], xy.y.clone())?;

    let mut w = T::new(vec![xy.n + 1, 1], vec![0.0; (xy.n + 1) as usize])?;

    let now = Instant::now();

    w = logistic_regression(&x, &y, w, l, e).unwrap();
    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);

    let x_test = T::new(vec![xy.m_test, xy.n], xy.x_test.clone())?;
    let y_test = T::new(vec![xy.m_test, 1], xy.y_test.clone())?;

    let predictions = predict_logistic(&x_test, &w)?;

    let mut correct = 0;
    let total = xy.m_test as usize;
    let predictions_data = predictions.get_data();
    let y_test_data = y_test.get_data();

    for i in 0..total {
        let pred = predictions_data[i];
        let actual = y_test_data[i];
        if (pred - actual).abs() < 1e-10 {
            correct += 1;
        }
    }

    let accuracy = (correct as f64) / (total as f64) * 100.0;
    println!("\nResults:");
    println!("Total samples: {}", total);
    println!("Correct predictions: {}", correct);
    println!("Accuracy: {:.2?}%", accuracy);

    Ok(())
}
