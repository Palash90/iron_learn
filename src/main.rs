use cust::error::CudaResult;
use cust::memory::DeviceBuffer;
use cust::module::Module;
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use iron_learn::{Data, GLOBAL_CONTEXT, Tensor, init_context};
use std::{env, fs};
use std::time::Instant;

// ---- Helpers: normalization + bias ----

#[derive(Clone, Debug)]
struct NormStats {
    means: Vec<f64>,
    stds: Vec<f64>,
}

fn compute_norm_stats(x: &[f64], rows: usize, cols: usize) -> NormStats {
    let mut means = vec![0.0; cols];
    let mut stds = vec![0.0; cols];

    // means
    for r in 0..rows {
        let base = r * cols;
        for c in 0..cols {
            means[c] += x[base + c];
        }
    }
    for c in 0..cols {
        means[c] /= rows as f64;
    }

    // stds
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
            stds[c] = 1.0; // avoid divide-by-zero
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
        // copy features
        let src_base = r * cols;
        let dst_base = r * (cols + 1);
        out[dst_base..dst_base + cols].copy_from_slice(&x[src_base..src_base + cols]);
        // bias = 1.0
        out[dst_base + cols] = 1.0;
    }
    out
}

// ---- Prediction (uses same stats and bias as training) ----

fn predict_logistic_gpu(
    module: &Module,
    stream: &Stream,
    x: &Vec<f64>,           // raw test features
    rows: usize,
    cols: usize,            // feature columns (without bias)
    w: &DeviceBuffer<f64>,  // weights on GPU (length cols+1)
    stats: &NormStats,      // training stats for normalization
) -> CudaResult<Vec<f64>> {
    // Preprocess test: normalize + add bias
    let x_norm = normalize_with_stats(x, rows, cols, stats);
    let x_bias = add_bias_column(&x_norm, rows, cols);

    // Device buffers
    let d_X = DeviceBuffer::from_slice(&x_bias)?;
    let d_lines = DeviceBuffer::<f64>::zeroed(rows)?;
    let d_prob = DeviceBuffer::<f64>::zeroed(rows)?;
    let d_pred = DeviceBuffer::<f64>::zeroed(rows)?;

    // Kernels
    let gemv = module.get_function("gemvRowMajor")?;
    let sigmoid = module.get_function("sigmoidKernel")?;
    let threshold = module.get_function("thresholdKernel")?;

    // z = X * w
    let block1d = (256, 1, 1);
    let grid_rows = ((rows as u32) + 255) / 256;
    unsafe {
        launch!(gemv<<<(grid_rows,1,1), block1d, 0, stream>>>(
            d_X.as_device_ptr(),
            w.as_device_ptr(),
            d_lines.as_device_ptr(),
            rows as i32,
            (cols + 1) as i32 // includes bias
        ))?;
    }

    // prob = sigmoid(z)
    unsafe {
        launch!(sigmoid<<<(grid_rows,1,1), block1d, 0, stream>>>(
            d_lines.as_device_ptr(),
            d_prob.as_device_ptr(),
            rows as i32
        ))?;
    }

    // pred = threshold(prob, 0.5)
    unsafe {
        launch!(threshold<<<(grid_rows,1,1), block1d, 0, stream>>>(
            d_prob.as_device_ptr(),
            d_pred.as_device_ptr(),
            rows as i32
        ))?;
    }

    stream.synchronize()?;

    // Copy back
    let mut predictions = vec![0.0f64; rows];
    d_pred.copy_to(&mut predictions)?;
    Ok(predictions)
}

// ---- Training orchestrator (uses gemv for logits and gradGemvXT for gradient) ----

pub fn run_ml_cuda() -> cust::error::CudaResult<()> {
    let l: f64 = 0.001;    // learning rate
    let e: usize = 10000;  // epochs

    let contents = fs::read_to_string("data.json").expect("Failed to read data.json");
    let data: Data = serde_json::from_str(&contents).unwrap();
    let Data { logistic, .. } = data;

    let rows = logistic.m as usize;
    let cols = logistic.n as usize;

    // Compute normalization stats on training data
    let stats = compute_norm_stats(&logistic.x, rows, cols);
    let x_norm = normalize_with_stats(&logistic.x, rows, cols, &stats);
    let x_bias = add_bias_column(&x_norm, rows, cols); // training with bias

    // Load PTX module
    let ptx = include_str!("../kernels/gradient_descent.ptx");
    let module = Module::from_ptx(ptx, &[])?;

    // Retrieve kernels
    let gemv = module.get_function("gemvRowMajor")?;
    let sigmoid = module.get_function("sigmoidKernel")?;
    let vector_sub = module.get_function("vectorSub")?;
    let scale_vec = module.get_function("scaleVector")?;
    let update_w = module.get_function("updateWeights")?;
    // Gradient via X^T · loss
    let grad_xt = module.get_function("gradGemvXT")?; // ensure this kernel exists in your PTX

    // Allocate device buffers
    let d_X = DeviceBuffer::from_slice(&x_bias)?;                // (rows × (cols+1))
    let d_y = DeviceBuffer::from_slice(&logistic.y)?;            // (rows)
    let d_w = DeviceBuffer::from_slice(&vec![0.0f64; cols + 1])?; // weights incl. bias
    let d_lines = DeviceBuffer::<f64>::zeroed(rows)?;
    let d_prob = DeviceBuffer::<f64>::zeroed(rows)?;
    let d_loss = DeviceBuffer::<f64>::zeroed(rows)?;
    let d_grad = DeviceBuffer::<f64>::zeroed(cols + 1)?;      // gradient incl. bias

    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Kernel launch params
    let block1d = (256, 1, 1);
    let grid_rows = ((rows as u32) + 255) / 256;
    let grid_cols = (((cols + 1) as u32) + 255) / 256;

    let start = Instant::now();

    // Training loop
    for i in 0..e {
        // 1) lines = X * w   (gemv over rows)
        unsafe {
            launch!(gemv<<<(grid_rows,1,1), block1d, 0, stream>>>(
                d_X.as_device_ptr(),
                d_w.as_device_ptr(),
                d_lines.as_device_ptr(),
                rows as i32,
                (cols + 1) as i32
            ))?;
        }

        // 2) prob = sigmoid(lines)
        unsafe {
            launch!(sigmoid<<<(grid_rows,1,1), block1d, 0, stream>>>(
                d_lines.as_device_ptr(),
                d_prob.as_device_ptr(),
                rows as i32
            ))?;
        }

        // 3) loss = prob - y
        unsafe {
            launch!(vector_sub<<<(grid_rows,1,1), block1d, 0, stream>>>(
                d_prob.as_device_ptr(),
                d_y.as_device_ptr(),
                d_loss.as_device_ptr(),
                rows as i32
            ))?;
        }

        // 4) grad = X^T * loss   (accumulate per-feature, incl. bias)
        unsafe {
            launch!(grad_xt<<<(grid_cols,1,1), block1d, 0, stream>>>(
                d_X.as_device_ptr(),     // (rows × (cols+1)), row-major
                d_loss.as_device_ptr(),  // (rows)
                d_grad.as_device_ptr(),  // (cols+1)
                rows as i32,
                (cols + 1) as i32
            ))?;
        }

        // 5) scale gradient by (l / m)
        unsafe {
            launch!(scale_vec<<<(grid_cols,1,1), block1d, 0, stream>>>(
                d_grad.as_device_ptr(),
                l / rows as f64,
                (cols + 1) as i32
            ))?;
        }

        // 6) update weights: w -= grad
        unsafe {
            launch!(update_w<<<(grid_cols,1,1), block1d, 0, stream>>>(
                d_w.as_device_ptr(),
                d_grad.as_device_ptr(),
                (cols + 1) as i32
            ))?;
        }

        stream.synchronize()?;

        if i % 500 == 0 {
            // Optional: log training loss on host (cross-entropy)
            println!("Iteration {} complete", i);
        }
    }

    let duration = start.elapsed();
    println!("GPU Logistic Regression Training Time: {:.2?}", duration);

    // Copy final weights back
    let mut w_host = vec![0.0f64; cols + 1];
    d_w.copy_to(&mut w_host)?;
    println!("Final weights (first 10): {:?}", &w_host[..w_host.len().min(10)]);

    // Predict on test (same preprocessing)
    let y_test = Tensor::new(vec![logistic.m_test, 1], logistic.y_test.clone()).unwrap();
    let preds = predict_logistic_gpu(
        &module,
        &stream,
        &logistic.x_test.clone(),
        logistic.m_test as usize,
        cols,                // features without bias
        &d_w,                // weights with bias
        &stats,              // training normalization stats
    )?;

    // Accuracy: direct comparison (preds are 0/1)
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