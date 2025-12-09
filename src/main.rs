//! Iron Learn - Example Application
//!
//! Demonstrates usage of the machine learning library with both CPU and GPU backends.
//!
//! ## Usage
//!
//! ```bash
//! cargo run [--release]           # Auto-detect GPU (default)
//! cargo run cpu [lr] [epochs]     # CPU mode with optional hyperparameters
//! cargo run gpu [lr] [epochs]     # GPU mode with optional hyperparameters
//! ```
//!
//! ## Arguments
//!
//! - `mode`: `cpu` for CPU only (default: auto-detect with CPU fallback)
//! - `learning_rate`: Learning rate for optimization (default: 0.01)
//! - `epochs`: Number of training iterations (default: 10000)
//! - `data_file`: Path to JSON data file (default: data.json)

use cust::prelude::Module;
use cust::stream::Stream;
use cust::stream::StreamFlags;
use iron_learn::run_linear_cuda;
use iron_learn::run_logistics_cuda;
use iron_learn::{init_context, run_linear, run_logistic, GpuTensor, CpuTensor, GLOBAL_CONTEXT};
use std::env;
use std::time::Instant;

/// Parse command-line arguments and configure the application
///
/// Handles GPU detection, hyperparameter parsing, and data file specification.
/// Falls back to sensible defaults if arguments are missing or invalid.
fn init() {
    let args: Vec<String> = env::args().collect();

    // Parse GPU mode (default: attempt GPU)
    let gpu_enabled: bool = if args.len() > 1 {
        args[1] != "cpu"
    } else {
        true
    };

    // Parse hyperparameters with defaults
    let learning_rate = args
        .get(2)
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.01);
    let epochs = args
        .get(3)
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(10000);
    let data_path = args
        .get(4)
        .map(|s| s.to_owned())
        .unwrap_or_else(|| "data.json".to_owned());

    // Attempt GPU initialization if requested
    if gpu_enabled {
        match cust::quick_init() {
            Ok(context) => {
                eprintln!("✓ GPU initialization successful");

                let ptx = include_str!("../kernels/gpu_kernels.ptx");
                let module =
                    Module::from_ptx(ptx, &[]).expect("CUDA module could not be initiated");

                let stream = match Stream::new(StreamFlags::NON_BLOCKING, None) {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!("Error creating stream: {}", e);
                        return;
                    }
                };

                init_context(
                    "Iron Learn",
                    5,
                    data_path,
                    learning_rate,
                    epochs,
                    true,
                    Some(context),
                    Some(module),
                    Some(stream),
                );
            }
            Err(e) => {
                eprintln!("⚠ GPU initialization failed: {}. Using CPU mode.", e);
                init_context(
                    "Iron Learn",
                    5,
                    data_path,
                    learning_rate,
                    epochs,
                    false,
                    None,
                    None,
                    None,
                );
            }
        }
    } else {
        init_context(
            "Iron Learn",
            5,
            data_path,
            learning_rate,
            epochs,
            false,
            None,
            None,
            None,
        );
    }
}

fn greet(ctx: &iron_learn::AppContext) {
    println!("\n╔════════════════════════════════╗");
    println!("║ {} v{}", ctx.app_name, ctx.version);
    println!("║ Mode: {}", if ctx.gpu_enabled { "GPU" } else { "CPU" });
    println!("║ Learning Rate: {}", ctx.learning_rate);
    println!("║ Epochs: {}", ctx.epochs);
    println!("║ Data Path: {}", ctx.data_path);
    println!("╚════════════════════════════════╝\n");
}

fn main() {
    init();

    let ctx = GLOBAL_CONTEXT.get().expect("Context not initialized");
    greet(ctx);

    // Execute appropriate training pipeline
    if ctx.gpu_enabled {
        println!("Running GPU-based training...\n");
        
        let now = Instant::now();
        let _ = run_linear::<GpuTensor<f64>>();
        let elapsed = now.elapsed();
        println!("Linear Regression completed in {:.4?}", elapsed);

        let now = Instant::now();
        let _ = run_logistic::<GpuTensor<f64>>();
        let elapsed = now.elapsed();
        println!("Logistic Regression completed in {:.4?}", elapsed);


        let now = Instant::now();
        let _ = run_linear_cuda();
        let elapsed = now.elapsed();
        println!("Old linear Regression completed in {:.4?}", elapsed);

        let now = Instant::now();
        let _ = run_logistics_cuda();
        let elapsed = now.elapsed();
        println!("Old logistic Regression completed in {:.4?}", elapsed);

        println!("\n✓ All training tasks completed");
    } else {
        println!("Running CPU-based training...\n");
        let _ = run_linear::<CpuTensor<f64>>();
        let _ = run_logistic::<CpuTensor<f64>>();
        println!("\n✓ All training tasks completed");
    }
}
