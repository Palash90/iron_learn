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

use iron_learn::{
    init_context, run_linear, run_linear_cuda, run_logistic, run_logistics_cuda,
    run_neural_network, GpuTensor, GLOBAL_CONTEXT,
};
use std::{env, vec};

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
                init_context(
                    "Iron Learn",
                    5,
                    data_path,
                    learning_rate,
                    epochs,
                    true,
                    Some(context),
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

    // run_neural_network();

    let t = GpuTensor::new(vec![1, 2], vec![1, 2]);
    let t2 = GpuTensor::new(vec![1, 2], vec![1, 2]);

    println!("t and t2 are equal: {}", t == t2);

    // Execute appropriate training pipeline
    if ctx.gpu_enabled {
        match run_linear_cuda() {
            Ok(_) => println!("\n✓ Training completed successfully"),
            Err(e) => eprintln!("\n✗ Training failed: {}", e),
        }
        match run_logistics_cuda() {
            Ok(_) => println!("\n✓ Training completed successfully"),
            Err(e) => eprintln!("\n✗ Training failed: {}", e),
        }
    } else {
        println!("Running CPU-based training...\n");
        run_linear();
        run_logistic();
        println!("\n✓ All training tasks completed");
    }
}
