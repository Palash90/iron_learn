#[cfg(feature = "cuda")]
use crate::init_gpu;

use clap::Parser;

use super::contexts::init_context;
use super::contexts::AppContext;
use super::contexts::GLOBAL_CONTEXT;
use crate::examples::types::IronLearnArgs;

/// Initialize the application runtime and global context.
///
/// This function performs the following high-level steps:
///
/// 1. Parse command-line arguments into `Args` (uses `clap`).
/// 2. Resolve the requested weight initialization distribution into
///    `DistributionType` (e.g. `Normal`, `Xavier`, `Uniform`, `He`).
/// 3. Decide whether to enable GPU execution based on the `--cpu` flag
///    and whether the crate was built with the `cuda` feature.
/// 4. If GPU execution is enabled and the binary was compiled with
///    the `cuda` feature, attempt to initialize CUDA (via `cust`) and
///    cuBLAS. On success it will create a CUDA module and stream,
///    initialize the global application context via `init_context`,
///    and call `init_gpu` to attach GPU resources.
/// 5. If GPU initialization fails (or CUDA support is not compiled in),
///    the function falls back to CPU mode and initializes the context
///    accordingly.
/// 6. Finally, the function retrieves the initialized `GLOBAL_CONTEXT`
///    and prints a friendly status banner by calling `greet`.
///
/// Notes:
/// - This function does not return a value; it performs global side
///   effects (initializing `GLOBAL_CONTEXT` and printing to stdout).
/// - If CUDA initialization fails, the function prints an error and
///   continues in CPU mode rather than aborting the process.
///
/// Example CLI usage:
///
/// ```text
/// cargo run -- --name my_run --lr 0.01 --epochs 1000 --data-file data/neural_net.json
/// ```
pub fn init_runtime() -> &'static AppContext {
    let args = IronLearnArgs::parse();
    let gpu_enabled = !args.cpu;

    if gpu_enabled {
        {
            #[cfg(feature = "cuda")]
            {
                match init_gpu() {
                    Ok(_) => {
                        println!("✓ GPU initialization successful");

                        init_context("Iron Learn", 6, true, args);
                    }
                    Err(e) => {
                        eprintln!("⚠ GPU initialization failed: {}", e);
                        init_context("Iron Learn", 6, false, args);
                    }
                }
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            eprintln!("⚠ GPU support not compiled. Using CPU mode.");
            init_context("Iron Learn", 6, false, args);
        }
    } else {
        init_context("Iron Learn", 6, gpu_enabled, args);
    }

    let ctx = GLOBAL_CONTEXT.get().expect("Context not initialized");
    greet(ctx);
    ctx
}

/// Print a concise, human-readable summary of the current application
/// context to stdout.
///
/// This helper formats several important runtime properties from the
/// provided `AppContext` instance so users can verify the active
/// configuration (application name/version, selected model name,
/// execution mode, learning rate, number of epochs, etc.).
///
/// Parameters
/// - `ctx`: Reference to the `AppContext` previously created by
///   `init_context`. The function assumes the context has been
///   initialized and will read fields such as `app_name`, `version`,
///   `name`, `gpu_enabled`, `learning_rate`, `lr_adjust`, `epochs`,
///   `hidden_layer_length`, `data_path`, `weights_path`,
///   `monitor_interval`, and `sleep_time`.
///
/// This function is purely informational and has no side effects
/// beyond writing to stdout.
fn greet(ctx: &AppContext) {
    println!("\n╔═══════════════════════════════════════╗");
    println!("║ {} v{}", ctx.app_name, ctx.version);
    println!("║ Name: {}", ctx.name);
    println!("║ Mode: {}", if ctx.gpu_enabled { "GPU" } else { "CPU" });
    println!("║ Learning Rate: {}", ctx.learning_rate);
    println!("║ Learning Rate Adjustment: {}", ctx.lr_adjust);
    println!("║ Epochs: {}", ctx.epochs);
    println!("║ Hidden Layers: {}", ctx.hidden_layer_length);
    println!("║ Data Path: {}", ctx.data_path);
    println!("║ Monitor Interval: {}", ctx.monitor_interval);
    println!("║ Intermediate Sleep Time: {} seconds", ctx.sleep_time);
    println!("╚═══════════════════════════════════════╝\n");
}
