use std::time::Instant;

use crate::Tensor;
use crate::commons::add_bias_term;
use crate::commons::denormalize_features;
use crate::commons::normalize_features_mean_std;
use crate::examples::contexts::GLOBAL_CONTEXT;
use crate::examples::read_file::deserialize_data_double_precision;
use crate::gradient_descent::gradient_descent;
use crate::normalize_features;
use crate::tensor::math::TensorMath;


/// Predict outputs for `x` using linear model weights `w`.
pub fn predict_linear<T>(x: &T, w: &T) -> Result<T, String>
where
    T: Tensor<f64> + TensorMath<f64, MathOutput = T>,
{
    x.mul(w)
}

/// Train a linear regression model using gradient descent.
///
/// - `x`, `y`: training data
/// - `w`: initial weights (consumed)
/// - `l`: learning rate
/// - `e`: number of epochs
pub fn linear_regression<T>(x: &T, y: &T, w: T, l: f64, e: u32) -> Result<T, String>
where
    T: Tensor<f64> + TensorMath<f64, MathOutput = T>,
{
    let x_with_bias = add_bias_term(x)?;
    let mut weight = w;
    for _ in 0..(e - 1) {
        weight = gradient_descent(&x_with_bias, y, &weight, l, false).unwrap();
    }

    Ok(gradient_descent(&x_with_bias, y, &weight, l, false).unwrap())
}

/// Run linear regression using configuration from the global context.
///
/// The function loads data, performs normalization, trains a linear model,
/// evaluates on test data (if present), and prints MSE and RMSE. Returns
/// `Ok(())` on success or an error string.
pub fn run_linear<T>() -> Result<(), String>
where
    T: Tensor<f64> + TensorMath<f64, MathOutput = T>,
{
    let l = GLOBAL_CONTEXT
        .get()
        .ok_or("GLOBAL_CONTEXT not initialized")?
        .learning_rate;
    let e = GLOBAL_CONTEXT.get().unwrap().epochs;
    let data_path = &GLOBAL_CONTEXT.get().unwrap().data_path;

    let xy = deserialize_data_double_precision(data_path)
        .map_err(|e| format!("Data deserialization error: {}", e))?;

    print!("\nLinear Regression\n");
    print!("Number of examples (m): {}\n", xy.m);
    print!("Number of features (n): {}\n", xy.n);
    print!("Total X values length: {}\n", xy.x.len());
    print!("Total Y values length: {}\n", xy.y.len());

    let x = T::new(vec![xy.m, xy.n], xy.x.clone())?;
    let y = T::new(vec![xy.m, 1], xy.y.clone())?;

    let (x, x_mean, x_std) = normalize_features_mean_std(&x);
    let (y, y_mean, y_std) = normalize_features_mean_std(&y);

    let mut w = T::new(vec![xy.n + 1, 1], vec![0.0; (xy.n + 1) as usize])?;

    let now = Instant::now();

    w = linear_regression(&x, &y, w, l, e).unwrap();

    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);

    if xy.m_test == 0 {
        println!("\nNo test data available for prediction.");
        return Ok(());
    }

    let x_test = T::new(vec![xy.m_test, xy.n], xy.x_test.clone())?;
    let y_test = T::new(vec![xy.m_test, 1], xy.y_test.clone())?;

    let x_test = normalize_features(&x_test, &x_mean, &x_std);

    let predictions = predict_linear(&x_test, &w)?;

    let predictions = denormalize_features(&predictions, &y_mean, &y_std);

    let mut total_squared_error = 0.0;
    let total = xy.m_test as usize;
    let predictions_data = predictions.get_data();
    let y_test_data = y_test.get_data();

    for i in 0..total {
        let pred = predictions_data[i];
        let actual = y_test_data[i];
        let error = pred - actual;
        total_squared_error += error * error;
    }

    let mse = total_squared_error / (total as f64);
    println!("\nResults:");
    println!("Total test samples: {}", total);
    println!("Mean Squared Error: {:.4}", mse);
    println!("Root MSE: {:.4}", mse.sqrt());

    Ok(())
}
