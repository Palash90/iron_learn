use crate::{linear_regression, logistic_regression, predict_linear, predict_logistic};
use crate::{Tensor, GLOBAL_CONTEXT};
use crate::tensor_commons::TensorOps;
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct XY {
    pub m: u32,
    pub n: u32,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub m_test: u32,
    pub x_test: Vec<f64>,
    pub y_test: Vec<f64>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Data {
    pub linear: XY,
    pub logistic: XY,
    pub neural_network: XY,
}

pub fn run_logistic<T: TensorOps<f64>>() -> Result<(), String> {
    let l = GLOBAL_CONTEXT.get().ok_or("GLOBAL_CONTEXT not initialized")?.learning_rate;
    let e = GLOBAL_CONTEXT.get().unwrap().epochs;
    let data_path = &GLOBAL_CONTEXT.get().unwrap().data_path;

    let Data { logistic: xy, .. } = crate::read_file::deserialize_data(&data_path).map_err(|e| format!("Data deserialization error: {}", e))?;

    print!("\nLogistic Regression\n");
    print!("Number of examples (m): {}\n", xy.m);
    print!("Number of features (n): {}\n", xy.n);
    print!("Total X values length: {}\n", xy.x.len());
    print!("Total Y values length: {}\n", xy.y.len());

    // Sanity checks: ensure x and y lengths match m and n
    let expected_x_len = (xy.m * xy.n) as usize;
    if xy.x.len() != expected_x_len {
        return Err(format!(
            "Error: xy.x.len() = {}, but expected m * n = {}. Aborting logistic run.",
            xy.x.len(),
            expected_x_len
        ));
    }
    if xy.y.len() != xy.m as usize {
        return Err(format!(
            "Error: xy.y.len() = {}, but expected m = {}. Aborting logistic run.",
            xy.y.len(),
            xy.m
        ));
    }

    // Use T::new (assumed trait method) instead of Tensor::new
    let x = T::new(vec![xy.m, xy.n], xy.x.clone())?;
    let y = T::new(vec![xy.m, 1], xy.y.clone())?;

    // Initialize weights including bias term (one extra weight)
    let mut w = T::new(vec![xy.n + 1, 1], vec![0.0; (xy.n + 1) as usize])?;

    let now = Instant::now();
    let iter10 = Instant::now();

    for i in 0..e {
        if i % 10 == 0 {
            print!("Iteration: {} took {:.2?} \n", i, iter10.elapsed());
        }

        // Must handle the Result<T, String> returned by logistic_regression
        w = logistic_regression(&x, &y, &w, l)?;
    }
    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);

    // Initialize test data (no need to add bias here, predict_logistic will handle that)
    let x_test = T::new(vec![xy.m_test, xy.n], xy.x_test.clone())?;
    let y_test = T::new(vec![xy.m_test, 1], xy.y_test.clone())?;

    // Make predictions
    let predictions = predict_logistic(&x_test, &w)?;

    // Calculate accuracy
    let mut correct = 0;
    let total = xy.m_test as usize;
    let predictions_data = predictions.get_data(); // Assumed trait method
    let y_test_data = y_test.get_data();         // Assumed trait method

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
    println!("Accuracy: {:.2}%", accuracy);
    
    Ok(()) // Return success
}

pub fn run_linear<T: TensorOps<f64>>() -> Result<(), String> {
    let l = GLOBAL_CONTEXT.get().ok_or("GLOBAL_CONTEXT not initialized")?.learning_rate;
    let e = GLOBAL_CONTEXT.get().unwrap().epochs;
    let data_path = &GLOBAL_CONTEXT.get().unwrap().data_path;

    let Data { linear: xy, .. } = crate::read_file::deserialize_data(data_path).map_err(|e| format!("Data deserialization error: {}", e))?;
    
    print!("\nLinear Regression\n");
    print!("Number of examples (m): {}\n", xy.m);
    print!("Number of features (n): {}\n", xy.n);
    print!("Total X values length: {}\n", xy.x.len());
    print!("Total Y values length: {}\n", xy.y.len());

    // Use T::new (assumed trait method) instead of Tensor::new
    let x = T::new(vec![xy.m, xy.n], xy.x.clone())?;
    let y = T::new(vec![xy.m, 1], xy.y.clone())?;

    // Initialize weights (include bias term: n + 1)
    let mut w = T::new(vec![xy.n + 1, 1], vec![0.0; (xy.n + 1) as usize])?;

    let now = Instant::now();

    for _ in 0..e {
        // Must handle the Result<T, String> returned by linear_regression
        w = linear_regression(&x, &y, &w, l)?;
    }

    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);

    // Skip prediction if there's no test data
    if xy.m_test == 0 {
        println!("\nNo test data available for prediction.");
        return Ok(());
    }

    // Initialize test data
    let x_test = T::new(vec![xy.m_test, xy.n], xy.x_test.clone())?;
    let y_test = T::new(vec![xy.m_test, 1], xy.y_test.clone())?;

    // Make predictions using the trained weights
    let predictions = predict_linear(&x_test, &w)?;

    // Calculate Mean Squared Error
    let mut total_squared_error = 0.0;
    let total = xy.m_test as usize;
    let predictions_data = predictions.get_data(); // Assumed trait method
    let y_test_data = y_test.get_data();         // Assumed trait method

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

    Ok(()) // Return success
}