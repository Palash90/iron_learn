use cust::error::CudaResult;
use cust::launch;
use cust::memory::{CopyDestination, DeviceBuffer};
use cust::module::Module;
use cust::stream::{Stream, StreamFlags};
use iron_learn::*;
use std::env;
use std::fs;
use std::time::Instant;

pub fn run_ml() {
    let l = 0.001;

    let e = 5000;

    let contents = fs::read_to_string("data.json").expect("Should have been able to read the file");

    // Parse once and destructure into `linear` and `logistic` datasets
    let data: Data = serde_json::from_str(&contents).unwrap();
    let Data {
        linear: xy,
        logistic,
    } = data;

    // Run linear regression using the extracted function
    run_linear(&xy, l, e);

    // Call the extracted function to run logistic regression using the `logistic` dataset
    run_logistic(&logistic, l, e);
}

use cust::prelude::*;

/// Normalize features column-wise to zero mean and unit variance.
/// Input: flat Vec<f64> of shape [rows * cols], row-major.
/// Output: normalized Vec<f64> of same shape.
pub fn normalize_features_vec(x: &Vec<f64>, rows: usize, cols: usize) -> Vec<f64> {
    let mut normalized = vec![0.0; rows * cols];

    for c in 0..cols {
        // Extract column
        let mut col_vals = Vec::with_capacity(rows);
        for r in 0..rows {
            col_vals.push(x[r * cols + c]);
        }

        // Compute mean and std
        let mean: f64 = col_vals.iter().sum::<f64>() / rows as f64;
        let var: f64 = col_vals.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / rows as f64;
        let std = var.sqrt();

        // Normalize
        for r in 0..rows {
            let idx = r * cols + c;
            normalized[idx] = if std > 1e-12 {
                (x[idx] - mean) / std
            } else {
                0.0
            };
        }
    }

    normalized
}

/// Add bias term (column of 1s) to features.
/// Input: Vec<f64> of shape [rows * cols].
/// Output: Vec<f64> of shape [rows * (cols+1)], with last column = 1.0.
pub fn add_bias_term_vec(x: &Vec<f64>, rows: usize, cols: usize) -> Vec<f64> {
    let mut with_bias = vec![0.0; rows * (cols + 1)];

    for r in 0..rows {
        // Copy original features
        for c in 0..cols {
            with_bias[r * (cols + 1) + c] = x[r * cols + c];
        }
        // Add bias term
        with_bias[r * (cols + 1) + cols] = 1.0;
    }

    with_bias
}

// GPU version of predict_logistic
pub fn predict_logistic_gpu(
    module: &Module,
    stream: &Stream,
    x: &Vec<f64>, // raw features
    rows: usize,
    cols: usize,
    w: &DeviceBuffer<f64>, // weights already on GPU
) -> CudaResult<Vec<f64>> {
    // --- Preprocess on CPU before sending to GPU ---
    let x_normalized = normalize_features_vec(x, rows, cols);
    let x_with_bias = add_bias_term_vec(&x_normalized, rows, cols);

    // Copy features to GPU
    let mut d_X = DeviceBuffer::from_slice(&x_with_bias)?;
    let mut d_lines = DeviceBuffer::<f64>::zeroed(rows)?;
    let mut d_prob = DeviceBuffer::<f64>::zeroed(rows)?;
    let mut d_pred = DeviceBuffer::<f64>::zeroed(rows)?;

    // Retrieve kernels
    let gemv = module.get_function("gemvRowMajor")?;
    let sigmoid = module.get_function("sigmoidKernel")?;
    let threshold = module.get_function("thresholdKernel")?;

    // --- 1. z = X * w ---
    let block1d = (256, 1, 1);
    let grid_rows = ((rows as u32) + 255) / 256;
    unsafe {
        launch!(gemv<<<(grid_rows,1,1), block1d, 0, stream>>>(
            d_X.as_device_ptr(),
            w.as_device_ptr(),
            d_lines.as_device_ptr(),
            rows as i32,
            (cols+1) as i32 // +1 for bias
        ))?;
    }

    // --- 2. probabilities = sigmoid(z) ---
    unsafe {
        launch!(sigmoid<<<(grid_rows,1,1), block1d, 0, stream>>>(
            d_lines.as_device_ptr(),
            d_prob.as_device_ptr(),
            rows as i32
        ))?;
    }

    // --- 3. predictions = threshold(probabilities, 0.5) ---
    unsafe {
        launch!(threshold<<<(grid_rows,1,1), block1d, 0, stream>>>(
            d_prob.as_device_ptr(),
            d_pred.as_device_ptr(),
            rows as i32
        ))?;
    }

    stream.synchronize()?;

    // Copy predictions back to host
    let mut predictions = vec![0.0f64; rows];
    d_pred.copy_to(&mut predictions)?;

    Ok(predictions)
}

pub fn run_ml_cuda() -> cust::error::CudaResult<()> {
    let l: f32 = 0.001; // learning rate
    let e: usize = 5000; // epochs

    let contents = fs::read_to_string("data.json").expect("Failed to read data.json");
    let data: Data = serde_json::from_str(&contents).unwrap();
    let Data { logistic, .. } = data;

    let rows = logistic.m as usize;
    let cols = logistic.n as usize;

    // Load PTX module
    let ptx = include_str!("../kernels/gradient_descent.ptx");
    let module = Module::from_ptx(ptx, &[])?;

    // Retrieve kernels
    let matrix_mul = module.get_function("matrixMulKernel")?;
    let sigmoid = module.get_function("sigmoidKernel")?;
    let vector_sub = module.get_function("vectorSub")?;
    let scale_vec = module.get_function("scaleVector")?;
    let update_w = module.get_function("updateWeights")?;

    // Allocate device buffers
    let d_X = DeviceBuffer::from_slice(&logistic.x)?;
    let d_y = DeviceBuffer::from_slice(&logistic.y)?;
    let d_w = DeviceBuffer::from_slice(&vec![0.0f64; cols])?;
    let d_lines = DeviceBuffer::<f32>::zeroed(rows)?;
    let d_prediction = DeviceBuffer::<f32>::zeroed(rows)?;
    let d_loss = DeviceBuffer::<f32>::zeroed(rows)?;
    let d_grad = DeviceBuffer::<f32>::zeroed(cols)?;

    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Kernel launch params
    const TILE: u32 = 32;
    let block2d = (TILE, TILE, 1);
    let grid_x = ((cols as u32) + TILE - 1) / TILE;
    let grid_y = ((rows as u32) + TILE - 1) / TILE;
    let grid2d = (grid_x, grid_y, 1);

    let block1d = (256, 1, 1);
    let grid_rows = ((rows as u32) + 255) / 256;
    let grid_cols = ((cols as u32) + 255) / 256;

    let start = Instant::now();

    // Training loop
    for i in 0..e {
        // 1. lines = X * w
        unsafe {
            launch!(matrix_mul<<<grid2d, block2d, 0, stream>>>(
                d_X.as_device_ptr(),
                d_w.as_device_ptr(),
                d_lines.as_device_ptr(),
                rows as i32,
                cols as i32,
                1i32
            ))?;
        }

        // 2. prediction = sigmoid(lines)
        unsafe {
            launch!(sigmoid<<<(grid_rows,1,1), block1d, 0, stream>>>(
                d_lines.as_device_ptr(),
                d_prediction.as_device_ptr(),
                rows as i32
            ))?;
        }

        // 3. loss = prediction - y
        unsafe {
            launch!(vector_sub<<<(grid_rows,1,1), block1d, 0, stream>>>(
                d_prediction.as_device_ptr(),
                d_y.as_device_ptr(),
                d_loss.as_device_ptr(),
                rows as i32
            ))?;
        }

        // 4. gradient = X^T * loss
        unsafe {
            launch!(matrix_mul<<<(grid_x,1,1), block2d, 0, stream>>>(
                d_X.as_device_ptr(),
                d_loss.as_device_ptr(),
                d_grad.as_device_ptr(),
                cols as i32,
                rows as i32,
                1i32
            ))?;
        }

        // 5. scale gradient
        unsafe {
            launch!(scale_vec<<<(grid_cols,1,1), block1d, 0, stream>>>(
                d_grad.as_device_ptr(),
                l / rows as f32,
                cols as i32
            ))?;
        }

        // 6. update weights
        unsafe {
            launch!(update_w<<<(grid_cols,1,1), block1d, 0, stream>>>(
                d_w.as_device_ptr(),
                d_grad.as_device_ptr(),
                cols as i32
            ))?;
        }

        stream.synchronize()?;

        if i % 500 == 0 {
            println!("Iteration {} complete", i);
        }
    }

    let duration = start.elapsed();
    println!("GPU Logistic Regression Training Time: {:.2?}", duration);

    // Copy final weights back
    let mut w_host = vec![0.0f64; cols];
    d_w.copy_to(&mut w_host)?;
    println!("Final weights (first 10): {:?}", &w_host[..10]);

    // Initialize test data (no need to add bias here, predict_logistic will handle that)
    let y_test = Tensor::new(vec![logistic.m_test, 1], logistic.y_test.clone()).unwrap();

    // Make predictions (predict_logistic will add bias term internally)
    let predictions = predict_logistic_gpu(&module, &stream, &logistic.x_test.clone(), logistic.m_test as usize, 1, &d_w);

    // Calculate accuracy
    let mut correct = 0;
    let total = logistic.m_test as usize;

    for i in 0..total {
        let pred:f64 = match predictions.clone().unwrap().get(i) {
            Some(&val) => val,
            None => 0.0,
        };
        let actual = y_test.get_data()[i];

        if (pred - actual).abs() < 1e-10 {
            // Using small epsilon for floating point comparison
            correct += 1;
        }
    }

    let accuracy = (correct as f64) / (total as f64) * 100.0;
    println!("\nResults:");
    println!("Total samples: {}", total);
    println!("Correct predictions: {}", correct);
    println!("Accuracy: {:.2}%", accuracy);

    Ok(())
}

fn init() {
    let args: Vec<String> = env::args().collect();

    if args.len() > 1 && args[1] == "gpu" {
        // Initialize CUDA and create a default context
        let _context = cust::quick_init();

        match _context {
            Err(e) => {
                eprintln!("Failed to initialize CUDA: {}", e);
                init_context("Iron Learn", 1, false, None);
            }
            Ok(ctx) => {
                init_context("Iron Learn", 1, true, Some(ctx));
            }
        };
    } else {
        init_context("Iron Learn", 1, false, None);
    }
}

fn main() {
    init();
    let ctx = GLOBAL_CONTEXT.get().expect("Context not initialized");
    println!("Welcome to {} v{}.", ctx.app_name, ctx.version);
    match ctx.gpu_enabled {
        true => println!("GPU is enabled."),
        false => println!("GPU is not enabled."),
    }

    match run_ml_cuda() {
        Ok(_) => println!("CUDA ML run completed successfully."),
        Err(e) => eprintln!("CUDA ML run failed: {}", e),
    }
}
