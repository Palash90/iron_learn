//! # Application Context Module
//!
//! Provides global application state management and capability detection.
//!
//! Manages the single global application context containing:
//! - Application metadata (name, version)
//! - Training hyperparameters (learning rate, epochs)
//! - Data configuration (path to data file)
//! - GPU capability flags and CUDA context handle
//!
//! The context is initialized once at application startup and remains immutable
//! throughout the program lifetime using `OnceLock` for thread-safe access.

use std::sync::OnceLock;

use cust::{module::Module, stream::Stream};

use crate::cuda_tensor::CudaMemoryPool;

/// Global application context with training configuration and GPU capabilities
///
/// # Fields
///
/// * `app_name` - Application name/title
/// * `version` - Application version number
/// * `data_path` - Path to JSON data file
/// * `learning_rate` - Gradient descent learning rate
/// * `epochs` - Number of training iterations
/// * `gpu_enabled` - Flag indicating whether GPU acceleration is available
/// * `context` - CUDA context handle (None if GPU is not available)
///
/// # Example
///
/// ```rust,no_run
/// use iron_learn::{init_context, GLOBAL_CONTEXT};
///
/// // Initialize at startup with full configuration
/// init_context("MyApp", 1, "data.json".to_string(), 0.01, 10000, false, None, None, None);
///
/// // Access anywhere in the application
/// let ctx = GLOBAL_CONTEXT.get().unwrap();
/// if ctx.gpu_enabled {
///     println!("GPU acceleration available");
/// }
/// println!("Learning rate: {}", ctx.learning_rate);
/// ```
#[derive(Debug)]
pub struct AppContext {
    pub app_name: &'static str,
    pub version: u32,
    pub data_path: String,
    pub learning_rate: f64,
    pub epochs: u32,
    pub gpu_enabled: bool,
    pub context: Option<cust::context::Context>,
    pub module: Option<Module>,
    pub stream: Option<Stream>,
    pub pool: Option<CudaMemoryPool>
}

/// Global singleton instance of application context
///
/// Thread-safe access to the immutable application state initialized at startup.
/// Use `GLOBAL_CONTEXT.get()` to access the context after initialization.
pub static GLOBAL_CONTEXT: OnceLock<AppContext> = OnceLock::new();

/// Initialize the global application context
///
/// Must be called exactly once at application startup. Subsequent calls will fail silently.
/// This function captures all training configuration and GPU state for access throughout
/// the application lifetime.
///
/// # Arguments
///
/// * `app_name` - Application identifier (static lifetime)
/// * `version` - Version number for tracking
/// * `data_path` - Path to JSON data file for loading datasets
/// * `learning_rate` - Learning rate for gradient descent optimization
/// * `epochs` - Number of training iterations
/// * `gpu_enabled` - Whether GPU acceleration is enabled
/// * `context` - CUDA context handle (None for CPU-only mode)
///
/// # Example
///
/// ```rust,no_run
/// use iron_learn::init_context;
///
/// // Initialize with full configuration
/// init_context(
///     "IronLearn",
///     5,
///     "data.json".to_string(),
///     0.01,
///     10000,
///     false,
///     None,
///     None,
///     None
/// );
/// ```
pub fn init_context(
    app_name: &'static str,
    version: u32,
    data_path: String,
    learning_rate: f64,
    epochs: u32,
    gpu_enabled: bool,
    context: Option<cust::context::Context>,
    module: Option<Module>,
    stream: Option<Stream>
) {
    let pool =match context {
        Some(_) => Some(CudaMemoryPool::get_mem_pool()),
        None => None
    };

    let ctx = AppContext {
        app_name,
        version,
        data_path,
        learning_rate,
        epochs,
        gpu_enabled,
        context,
        module,
        stream,
        pool
    };
    match GLOBAL_CONTEXT.set(ctx) {
        Ok(_) => (),
        Err(_) => println!("AppContext has already been initialized!"),
    }
}
