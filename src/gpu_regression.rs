use crate::{Data, Tensor};
use cust::error::CudaResult;
use cust::memory::DeviceBuffer;
use cust::module::Module;
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::fs;
use std::time::Instant;

/// GPU-accelerated normalization and bias helpers for logistic regression preprocessing

#[derive(Clone, Debug)]
pub struct NormStats {
    means: Vec<f64>,
    stds: Vec<f64>,
}

pub fn compute_norm_stats(x: &[f64], rows: usize, cols: usize) -> NormStats {
    let mut means = vec![0.0; cols];
    let mut stds = vec![0.0; cols];

    // Compute column means
    for r in 0..rows {
        let base = r * cols;
        for c in 0..cols {
            means[c] += x[base + c];
        }
    }
    for c in 0..cols {
        means[c] /= rows as f64;
    }

    // Compute column standard deviations
    for r in 0..rows {
        let base = r * cols;
        for c in 0..cols {
            let d = x[base + c] - means[c];
            stds[c] += d * d;
        }
    }
    for c in 0..cols {
        stds[c] = (stds[c] / (rows as f64)).sqrt();
        if stds[c] == 0.0 {
            stds[c] = 1.0; // Prevent division by zero
        }
    }

    NormStats { means, stds }
}

fn normalize_with_stats(x: &[f64], rows: usize, cols: usize, stats: &NormStats) -> Vec<f64> {
    let mut out = vec![0.0; rows * cols];
    for r in 0..rows {
        let base = r * cols;
        for c in 0..cols {
            out[base + c] = (x[base + c] - stats.means[c]) / stats.stds[c];
        }
    }
    out
}

fn add_bias_column(x: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0; rows * (cols + 1)];
    for r in 0..rows {
        // Copy feature columns
        let src_base = r * cols;
        let dst_base = r * (cols + 1);
        out[dst_base..dst_base + cols].copy_from_slice(&x[src_base..src_base + cols]);
        // Append bias term
        out[dst_base + cols] = 1.0;
    }
    out
}

/// Prediction function for logistic regression using GPU-accelerated sigmoid
/// Uses training statistics for consistent normalization and learned bias term

pub fn predict_logistic_gpu(
    module: &Module,
    stream: &Stream,
    x: &Vec<f64>, // Raw test features (unnormalized)
    rows: usize,
    cols: usize,           // Feature column count (excludes bias)
    w: &DeviceBuffer<f64>, // Weight vector on GPU (includes bias term)
    stats: &NormStats,     // Training normalization statistics
) -> CudaResult<Vec<f64>> {
    // Preprocess: normalize using training statistics, then add bias column
    let x_norm = normalize_with_stats(x, rows, cols, stats);
    let x_bias = add_bias_column(&x_norm, rows, cols);

    // Allocate GPU buffers
    let d_x = DeviceBuffer::from_slice(&x_bias)?;
    let d_lines = DeviceBuffer::<f64>::zeroed(rows)?;
    let d_prob = DeviceBuffer::<f64>::zeroed(rows)?;
    let d_pred = DeviceBuffer::<f64>::zeroed(rows)?;

    // Load GPU kernels
    let gemv = module.get_function("gemvRowMajor")?;
    let sigmoid = module.get_function("sigmoidKernel")?;
    let threshold = module.get_function("thresholdKernel")?;

    // Compute logits: z = X·w
    let block1d = (256, 1, 1);
    let grid_rows = ((rows as u32) + 255) / 256;
    unsafe {
        launch!(gemv<<<(grid_rows,1,1), block1d, 0, stream>>>(
            d_x.as_device_ptr(),
            w.as_device_ptr(),
            d_lines.as_device_ptr(),
            rows as i32,
            (cols + 1) as i32 // includes bias
        ))?;
    }

    // Compute probabilities: prob = sigmoid(z)
    unsafe {
        launch!(sigmoid<<<(grid_rows,1,1), block1d, 0, stream>>>(
            d_lines.as_device_ptr(),
            d_prob.as_device_ptr(),
            rows as i32
        ))?;
    }

    // Threshold to binary predictions: pred = (prob >= 0.5) ? 1 : 0
    unsafe {
        launch!(threshold<<<(grid_rows,1,1), block1d, 0, stream>>>(
            d_prob.as_device_ptr(),
            d_pred.as_device_ptr(),
            rows as i32
        ))?;
    }

    stream.synchronize()?;

    // Transfer predictions from GPU to host memory
    let mut predictions = vec![0.0f64; rows];
    d_pred.copy_to(&mut predictions)?;
    Ok(predictions)
}

/// Main GPU training function for logistic regression
/// Implements gradient descent using CUDA kernels for efficient computation
/// Returns accuracy metrics on test set upon completion
pub fn run_ml_cuda() -> cust::error::CudaResult<()> {
    let l: f64 = 0.001; // Learning rate
    let e: usize = 10_000_000; // Training epochs

    let contents = fs::read_to_string("data.json").expect("Failed to read data.json");
    let data: Data = serde_json::from_str(&contents).unwrap();
    let Data { logistic, .. } = data;

    let rows = logistic.m as usize;
    let cols = logistic.n as usize;

    // Compute normalization statistics from training data
    let stats = compute_norm_stats(&logistic.x, rows, cols);
    let x_norm = normalize_with_stats(&logistic.x, rows, cols, &stats);
    let x_bias = add_bias_column(&x_norm, rows, cols); // Include bias column

    // Load compiled CUDA kernels from PTX module
    let ptx = include_str!("../kernels/gradient_descent.ptx");
    let module = Module::from_ptx(ptx, &[])?;

    // Retrieve kernel function references
    let gemv = module.get_function("gemvRowMajor")?;
    let sigmoid = module.get_function("sigmoidKernel")?;
    let vector_sub = module.get_function("vectorSub")?;
    let scale_vec = module.get_function("scaleVector")?;
    let update_w = module.get_function("updateWeights")?;
    // Compute gradient via matrix-transpose multiplication: grad = X^T · loss_vector
    let grad_xt = module.get_function("gradGemvXT")?;

    // Allocate GPU memory buffers
    let d_x = DeviceBuffer::from_slice(&x_bias)?; // Input features: rows × (cols+1)
    let d_y = DeviceBuffer::from_slice(&logistic.y)?; // Labels: rows
    let d_w = DeviceBuffer::from_slice(&vec![0.0f64; cols + 1])?; // Weights including bias
    let d_lines = DeviceBuffer::<f64>::zeroed(rows)?;
    let d_prob = DeviceBuffer::<f64>::zeroed(rows)?;
    let d_loss = DeviceBuffer::<f64>::zeroed(rows)?;
    let d_grad = DeviceBuffer::<f64>::zeroed(cols + 1)?; // Gradient including bias

    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Configure kernel launch parameters
    let block1d = (256, 1, 1);
    let grid_rows = ((rows as u32) + 255) / 256;
    let grid_cols = (((cols + 1) as u32) + 255) / 256;

    let start = Instant::now();

    // Main training loop: gradient descent iterations
    for i in 0..e {
        // 1) Forward pass: compute logits (z = X·w)
        unsafe {
            launch!(gemv<<<(grid_rows,1,1), block1d, 0, stream>>>(
                d_x.as_device_ptr(),
                d_w.as_device_ptr(),
                d_lines.as_device_ptr(),
                rows as i32,
                (cols + 1) as i32
            ))?;
        }

        // 2) Apply sigmoid activation to logits
        unsafe {
            launch!(sigmoid<<<(grid_rows,1,1), block1d, 0, stream>>>(
                d_lines.as_device_ptr(),
                d_prob.as_device_ptr(),
                rows as i32
            ))?;
        }

        // 3) Compute prediction error (loss = prediction - label)
        unsafe {
            launch!(vector_sub<<<(grid_rows,1,1), block1d, 0, stream>>>(
                d_prob.as_device_ptr(),
                d_y.as_device_ptr(),
                d_loss.as_device_ptr(),
                rows as i32
            ))?;
        }

        // 4) Compute gradient: grad = X^T · loss_vector (accumulated per feature)
        unsafe {
            launch!(grad_xt<<<(grid_cols,1,1), block1d, 0, stream>>>(
                d_x.as_device_ptr(),     // Feature matrix in row-major order
                d_loss.as_device_ptr(),  // Loss vector per sample
                d_grad.as_device_ptr(),  // Accumulated gradient per weight
                rows as i32,
                (cols + 1) as i32
            ))?;
        }

        // 5) Normalize gradient by learning rate and batch size (l / m)
        unsafe {
            launch!(scale_vec<<<(grid_cols,1,1), block1d, 0, stream>>>(
                d_grad.as_device_ptr(),
                l / rows as f64,
                (cols + 1) as i32
            ))?;
        }

        // 6) Update weights using gradient descent: w ← w - grad
        unsafe {
            launch!(update_w<<<(grid_cols,1,1), block1d, 0, stream>>>(
                d_w.as_device_ptr(),
                d_grad.as_device_ptr(),
                (cols + 1) as i32
            ))?;
        }

        stream.synchronize()?;

        if i % 500 == 0 {
            // Progress checkpoint: log iteration count every 500 iterations
            println!("Iteration {} complete", i);
        }
    }

    let duration = start.elapsed();
    println!("GPU Logistic Regression Training Time: {:.2?}", duration);

    // Transfer learned weights from GPU to host
    let mut w_host = vec![0.0f64; cols + 1];
    d_w.copy_to(&mut w_host)?;
    println!(
        "Final weights (first 10): {:?}",
        &w_host[..w_host.len().min(10)]
    );

    // Evaluate on test set using learned model and training preprocessing
    let y_test = Tensor::new(vec![logistic.m_test, 1], logistic.y_test.clone()).unwrap();
    let preds = predict_logistic_gpu(
        &module,
        &stream,
        &logistic.x_test.clone(),
        logistic.m_test as usize,
        cols,   // features without bias
        &d_w,   // weights with bias
        &stats, // training normalization stats
    )?;

    // Calculate accuracy by comparing predictions to ground truth labels
    let mut correct = 0;
    let total = logistic.m_test as usize;
    for i in 0..total {
        let pred = preds[i];
        let actual = y_test.get_data()[i];
        if pred == actual {
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
