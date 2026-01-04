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

use std::{sync::OnceLock, u32};

use crate::nn::DistributionType;

/// Global singleton instance of application context
///
/// Thread-safe access to the immutable application state initialized at startup.
/// Use `GLOBAL_CONTEXT.get()` to access the context after initialization.
pub static GLOBAL_CONTEXT: OnceLock<AppContext> = OnceLock::new();
use crate::examples::init::ExampleMode;

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
/// * `lr_adjust` - Flag indicating whether learning rate adjustment is enabled
/// * `hidden_layer_length` - Number of hidden layers in the neural network
/// * `weights_path` - Path to model weights file, the parameters file
/// * `monitor_interval` - The interval on how many epochs, the monitor should be called for internal state monitoring
/// * `sleep_time` - Sometimes the workload is high and CPU/GPU gets exhausted and machine generates a lot of heat, you may choose to let it cool down
/// * `name` - Name of the model. A similarly named directory must exist from execution path
/// * `restore` - Restore a network from model file.
/// * `distribution` - Weight initialization distribution
///

#[derive(Debug)]
pub struct AppContext {
    pub app_name: &'static str,
    pub version: u32,
    pub data_path: String,
    pub learning_rate: f64,
    pub epochs: u32,
    pub gpu_enabled: bool,
    pub lr_adjust: bool,
    pub hidden_layer_length: u32,
    pub weights_path: String,
    pub monitor_interval: usize,
    pub sleep_time: u64,
    pub name: String,
    pub restore: bool,
    pub distribution: DistributionType,
    pub example_mode: ExampleMode,
    pub predict_only: bool,
    pub resize: u32,
}

/// Initialize the global application context
///
/// Must be called exactly once at application startup. Subsequent calls will fail silently.
/// This function captures all training configuration and GPU state for access throughout
/// the application lifetime.
///
/// # Arguments
///
/// * `app_name` - Application name/title
/// * `version` - Application version number
/// * `data_path` - Path to JSON data file
/// * `learning_rate` - Gradient descent learning rate
/// * `epochs` - Number of training iterations
/// * `gpu_enabled` - Flag indicating whether GPU acceleration is available
/// * `context` - CUDA context handle (None if GPU is not available)
/// * `lr_adjust` - Flag indicating whether learning rate adjustment is enabled
/// * `hidden_layer_length` - Number of hidden layers in the neural network
/// * `weights_path` - Path to model weights file, the parameters file
/// * `monitor_interval` - The interval on how many epochs, the monitor should be called for internal state monitoring
/// * `sleep_time` - Sometimes the workload is high and CPU/GPU gets exhausted and machine generates a lot of heat, you may choose to let it cool down
/// * `name` - Name of the model. A similarly named directory must exist from execution path
/// * `restore` - Restore a network from model file.
/// * `distribution` - Weight initialization distribution
///
pub fn init_context(
    app_name: &'static str,
    version: u32,
    data_path: String,
    learning_rate: f64,
    epochs: u32,
    gpu_enabled: bool,
    lr_adjust: bool,
    hidden_layer_length: u32,
    weights_path: String,
    monitor_interval: usize,
    sleep_time: u64,
    name: String,
    restore: bool,
    distribution: DistributionType,
    example_mode: ExampleMode,
    predict_only: bool,
    resize: u32,
) {
    let ctx = AppContext {
        app_name,
        version,
        data_path,
        learning_rate,
        epochs,
        gpu_enabled,
        lr_adjust,
        hidden_layer_length,
        weights_path,
        monitor_interval,
        sleep_time,
        name,
        restore,
        distribution,
        example_mode,
        predict_only,
        resize,
    };
    match GLOBAL_CONTEXT.set(ctx) {
        Ok(_) => (),
        Err(_) => eprintln!("AppContext has already been initialized!"),
    }
}
