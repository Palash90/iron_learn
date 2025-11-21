/// Application entry point for Iron Learn ML library
///
/// This module handles:
/// - CUDA/GPU capability detection and initialization
/// - Global context setup with version and configuration information
/// - Orchestration of GPU-accelerated machine learning operations
///
/// The application accepts optional command-line arguments:
/// - "gpu": Enable GPU acceleration (requires CUDA installation)
/// - (default): Run without GPU acceleration (CPU only)

use std::env;

use iron_learn::{init_context, run_ml_cuda, GLOBAL_CONTEXT};

/// Initialize the application context with GPU capability detection
///
/// Attempts to initialize CUDA context if GPU mode is requested.
/// Falls back to CPU-only mode if GPU initialization fails or is not requested.
fn init() {
    let args: Vec<String> = env::args().collect();

    if args.len() > 1 && args[1] == "gpu" {
        // Attempt to initialize CUDA with automatic device detection
        let _context = cust::quick_init();

        match _context {
            Err(e) => {
                eprintln!("Failed to initialize CUDA: {}", e);
                eprintln!("Falling back to CPU-only mode...");
                init_context("Iron Learn", 1, false, None);
            }
            Ok(ctx) => {
                eprintln!("CUDA initialization successful.");
                init_context("Iron Learn", 1, true, Some(ctx));
            }
        };
    } else {
        // CPU-only mode
        init_context("Iron Learn", 1, false, None);
    }
}

/// Main application entry point
fn main() {
    init();
    let ctx = GLOBAL_CONTEXT.get().expect("Context not initialized");
    println!("Welcome to {} v{}.", ctx.app_name, ctx.version);
    match ctx.gpu_enabled {
        true => println!("GPU acceleration enabled."),
        false => println!("Running in CPU-only mode."),
    }

    // Execute GPU-accelerated machine learning pipeline
    match run_ml_cuda() {
        Ok(_) => println!("\n✓ ML operations completed successfully."),
        Err(e) => eprintln!("\n✗ ML operations failed: {}", e),
    }
}
