//! # Application Context Module
//!
//! Provides global application state management and capability detection.
//!
//! This module manages the single global application context containing:
//! - Application metadata (name, version)
//! - GPU capability flags
//! - CUDA context (if available)
//!
//! The context is initialized once at application startup and remains immutable
//! throughout the program lifetime using `OnceLock` for thread-safe access.

use std::sync::OnceLock;

/// Global application context holding configuration and GPU capabilities
///
/// # Fields
///
/// * `app_name` - Application name/title
/// * `version` - Application version number
/// * `gpu_enabled` - Flag indicating whether GPU acceleration is available
/// * `context` - CUDA context handle (None if GPU is not available)
///
/// # Example
///
/// ```rust,no_run
/// use iron_learn::{init_context, GLOBAL_CONTEXT};
///
/// // Initialize at startup
/// init_context("MyApp", 1, false, None);
///
/// // Access anywhere in the application
/// let ctx = GLOBAL_CONTEXT.get().unwrap();
/// if ctx.gpu_enabled {
///     println!("GPU acceleration available");
/// }
/// ```
#[derive(Debug)]
pub struct AppContext {
    pub app_name: &'static str,
    pub version: u32,
    pub gpu_enabled: bool,
    pub context: Option<cust::context::Context>,
}

/// Global singleton instance of application context
///
/// Thread-safe access to the immutable application state initialized at startup.
/// Use `GLOBAL_CONTEXT.get()` to access the context after initialization.
pub static GLOBAL_CONTEXT: OnceLock<AppContext> = OnceLock::new();

/// Initialize the global application context
///
/// Must be called exactly once at application startup. Subsequent calls will fail silently.
///
/// # Arguments
///
/// * `app_name` - Application name (static lifetime)
/// * `version` - Version number
/// * `gpu_enabled` - Whether GPU acceleration is enabled
/// * `context` - CUDA context handle (None for CPU-only mode)
///
/// # Panics
///
/// Will not panic, but will print a message if called more than once.
///
/// # Example
///
/// ```rust,no_run
/// use iron_learn::init_context;
///
/// // Initialize at program start
/// init_context("IronLearn", 1, true, None);
/// ```
pub fn init_context(
    app_name: &'static str,
    version: u32,
    gpu_enabled: bool,
    context: Option<cust::context::Context>,
) {
    let ctx = AppContext {
        app_name,
        version,
        gpu_enabled,
        context,
    };
    match GLOBAL_CONTEXT.set(ctx) {
        Ok(_) => (),
        Err(_) => println!("AppContext has already been initialized!"),
    }
}
