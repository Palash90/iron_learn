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

use std::collections::HashMap;
use std::ptr;
use std::sync::{Mutex, OnceLock};

use crate::cuda_tensor::CublasHandle;
use crate::cuda_tensor::CudaMemoryPool;
use cublas_sys::*;
use cust::prelude::Function;
use cust::{module::Module, stream::Stream};
use std::sync::Arc;
use std::time::Instant;

/// Global singleton instance of application context
///
/// Thread-safe access to the immutable application state initialized at startup.
/// Use `GLOBAL_CONTEXT.get()` to access the context after initialization.
pub static GLOBAL_CONTEXT: OnceLock<AppContext> = OnceLock::new();

pub static GPU_CONTEXT: OnceLock<GpuContext> = OnceLock::new();

static KERNEL_CONTEXT: OnceLock<Mutex<KernelMap>> = OnceLock::new();

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
}

#[derive(Debug)]
pub struct GpuContext {
    pub context: Option<cust::context::Context>,
    pub module: Option<Module>,
    pub stream: Option<Stream>,
    pub pool: Option<CudaMemoryPool>,
    pub cublas_handle: CublasHandle,
}

pub struct KernelMap {
    kernel_map: HashMap<String, Arc<Function<'static>>>,
}

impl GpuContext {
    pub fn get_function(&self, fn_name: &str) -> Arc<Function> {
        let mut guard = KERNEL_CONTEXT
            .get_or_init(|| {
                Mutex::new(KernelMap {
                    kernel_map: HashMap::new(),
                })
            })
            .lock()
            .unwrap();

        if let Some(f) = guard.kernel_map.get(fn_name) {
            return Arc::clone(f);
        }

        let now = Instant::now();
        let module = self.module.as_ref().expect("Module not found");

        match module.get_function(fn_name) {
            Ok(f) => {
                let f_static = unsafe { std::mem::transmute::<Function<'_>, Function<'static>>(f) };
                let shared_fn = Arc::new(f_static);

                guard
                    .kernel_map
                    .insert(fn_name.to_string(), Arc::clone(&shared_fn));
                shared_fn
            }
            Err(e) => {
                panic!("Error: {}, while getting function: {}", e, fn_name);
            }
        }
    }
}

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
) {
    let ctx = AppContext {
        app_name,
        version,
        data_path,
        learning_rate,
        epochs,
        gpu_enabled,
    };
    match GLOBAL_CONTEXT.set(ctx) {
        Ok(_) => (),
        Err(_) => eprintln!("AppContext has already been initialized!"),
    }
}

pub fn init_gpu(
    context: Option<cust::context::Context>,
    module: Option<Module>,
    stream: Option<Stream>,
    cublas_handle: Option<cublasHandle_t>,
) {
    let pool = match context {
        Some(_) => Some(CudaMemoryPool::get_mem_pool()),
        None => None,
    };

    let handle = match cublas_handle {
        Some(t) => t,
        _ => ptr::null_mut(),
    };

    let ctx = GpuContext {
        context,
        module,
        stream,
        pool,
        cublas_handle: CublasHandle { handle },
    };

    match GPU_CONTEXT.set(ctx) {
        Ok(_) => (),
        Err(_) => eprintln!("GpuContext has already been initialized!"),
    };
}
