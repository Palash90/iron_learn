use std::time::Instant;

use crate::{
    commons::{denormalize_features, normalize_features_mean_std},
    examples::{contexts::GLOBAL_CONTEXT, read_file::deserialize_data, types::Data},
    linear_regression::{linear_regression, predict_linear},
    logistic_regression::{logistic_regression, predict_logistic},
    normalize_features,
    numeric::FloatingPoint,
    tensor::math::TensorMath,
    Tensor,
};

/// Run linear regression using configuration from the global context.
///
/// The function loads data, performs normalization, trains a linear model,
/// evaluates on test data (if present), and prints MSE and RMSE. Returns
/// `Ok(())` on success or an error string.
pub fn run_linear<T, D>() -> Result<(), String>
where
    T: Tensor<D> + TensorMath<D, MathOutput = T>,
    D: FloatingPoint + From<f64>,
{
    let l = GLOBAL_CONTEXT
        .get()
        .ok_or("GLOBAL_CONTEXT not initialized")?
        .learning_rate;
    let e = GLOBAL_CONTEXT.get().unwrap().epochs;
    let data_path = &GLOBAL_CONTEXT.get().unwrap().data_path;

    let xy =
        deserialize_data(data_path).map_err(|e| format!("Data deserialization error: {}", e))?;

    print!("\nLinear Regression\n");
    print!("Number of examples (m): {}\n", xy.m);
    print!("Number of features (n): {}\n", xy.n);
    print!("Total X values length: {}\n", xy.x.len());
    print!("Total Y values length: {}\n", xy.y.len());

    let x = T::new(vec![xy.m, xy.n], xy.x.clone())?;
    let y = T::new(vec![xy.m, 1], xy.y.clone())?;

    let (x, x_mean, x_std) = normalize_features_mean_std(&x);
    let (y, y_mean, y_std) = normalize_features_mean_std(&y);

    let mut w = T::new(vec![xy.n + 1, 1], vec![D::zero(); (xy.n + 1) as usize])?;

    let now = Instant::now();

    w = linear_regression(&x, &y, w, D::from_f64(l), e).unwrap();

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

    let mut total_squared_error =D::zero();
    let total = xy.m_test as usize;
    let predictions_data = predictions.get_data();
    let y_test_data = y_test.get_data();

    for i in 0..total {
        let pred = predictions_data[i];
        let actual = y_test_data[i];
        let error = pred - actual;
        total_squared_error = total_squared_error + error * error;
    }

    let mse = total_squared_error / D::from_u32(total as u32);
    println!("\nResults:");
    println!("Total test samples: {}", total);
    println!("Mean Squared Error: {:.4}", mse);
    println!("Root MSE: {:.4}", mse.sqrt());

    Ok(())
}

/// Run logistic regression using configuration from the global context.
///
/// The function reads data and runtime settings from `GLOBAL_CONTEXT`,
/// trains a logistic model, evaluates on test data (if present), and
/// prints summary metrics. Returns `Ok(())` on success or an error string.
pub fn run_logistic<T, D>() -> Result<(), String>
where
    T: Tensor<D> + TensorMath<D, MathOutput = T>,
    D: FloatingPoint + From<f64>,
{
    let l = GLOBAL_CONTEXT
        .get()
        .ok_or("GLOBAL_CONTEXT not initialized")?
        .learning_rate;
    let e = GLOBAL_CONTEXT.get().unwrap().epochs;
    let data_path = &GLOBAL_CONTEXT.get().unwrap().data_path;

    let xy: Data<D> =
        deserialize_data(&data_path).map_err(|e| format!("Data deserialization error: {}", e))?;

    print!("\nLogistic Regression\n");
    print!("Number of examples (m): {}\n", xy.m);
    print!("Number of features (n): {}\n", xy.n);
    print!("Total X values length: {}\n", xy.x.len());
    print!("Total Y values length: {}\n", xy.y.len());

    let x = T::new(vec![xy.m, xy.n], xy.x.clone())?;
    let y = T::new(vec![xy.m, 1], xy.y.clone())?;

    let mut w = T::new(vec![xy.n + 1, 1], vec![D::zero(); (xy.n + 1) as usize])?;

    let now = Instant::now();

    w = logistic_regression(&x, &y, w, D::from_f64(l), e).unwrap();
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
        if (pred - actual).abs() < D::from_f64(1e-10) {
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
