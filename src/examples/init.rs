#[cfg(feature = "cuda")]
use crate::init_gpu;

use clap::Parser;

use crate::nn::DistributionType;

use super::contexts::init_context;
use super::contexts::AppContext;
use super::contexts::GLOBAL_CONTEXT;
use clap::ValueEnum;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
pub enum ExampleMode {
    /// Linear Regression
    Linear,
    /// Logistic Regression
    Logistic,
    /// Neural Network - Generic
    NeuralNet,
    /// Neural Network - XOR
    XorNeuralNet,
    /// Neural Network - Image
    ImageNeuralNet,
    /// Bigram Generator
    Bigram,
}

#[derive(Parser)]
#[command(name = "Iron Learn")]
#[command(name = "A Rust Machine Learning Library")]
struct Args {
    #[arg(long, short, default_value = "neural_net")]
    name: String,

    #[arg(long, short, default_value = "false")]
    cpu: bool,

    #[arg(long, short = 'x', default_value = "linear")]
    mode: ExampleMode,

    #[arg(long, short, default_value = "false")]
    restore: bool,

    #[arg(long, short, default_value = "0.01")]
    lr: f64,

    #[arg(long, short, default_value = "10001")]
    epochs: u32,

    #[arg(long, short, default_value = "data/neural_net.json")]
    data_file: String,

    #[arg(long, short, default_value = "false")]
    adjust_lr: bool,

    #[arg(long, short, default_value = "4")]
    internal_layers: u32,

    #[arg(long, short, default_value = "1000")]
    monitor_interval: usize,

    #[arg(long, short, default_value = "0")]
    sleep_time: u64,

    #[arg(long, short, default_value = "model.json")]
    parameters_path: String,

    #[arg(long, short = 'D', default_value = "Normal")]
    distribution: String,

    #[arg(long, default_value = "false")]
    predict_only: bool,

    #[arg(long, default_value = "0")]
    resize: u32,
}

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
    let args = Args::parse();
    let gpu_enabled = !args.cpu;

    let distribution = match args.distribution.as_str().to_uppercase().as_str() {
        "NORMAL" => DistributionType::Normal,
        "XAVIER" => DistributionType::Xavier,
        "UNIFORM" => DistributionType::Uniform,
        "He" => DistributionType::He,
        _ => DistributionType::Normal,
    };

    if gpu_enabled {
        {
            #[cfg(feature = "cuda")]
            {
                match init_gpu() {
                    Ok(_) => {
                        println!("✓ GPU initialization successful");

                        init_context(
                            "Iron Learn",
                            6,
                            args.data_file,
                            args.lr,
                            args.epochs,
                            true,
                            args.adjust_lr,
                            args.internal_layers,
                            args.parameters_path,
                            args.monitor_interval,
                            args.sleep_time,
                            args.name,
                            args.restore,
                            distribution,
                            args.mode,
                            args.predict_only,
                            args.resize,
                        );
                    }
                    Err(e) => {
                        eprintln!("⚠ GPU initialization failed: {}", e);
                        init_context(
                            "Iron Learn",
                            6,
                            args.data_file,
                            args.lr,
                            args.epochs,
                            false,
                            args.adjust_lr,
                            args.internal_layers,
                            args.parameters_path,
                            args.monitor_interval,
                            args.sleep_time,
                            args.name,
                            args.restore,
                            distribution,
                            args.mode,
                            args.predict_only,
                            args.resize,
                        );
                    }
                }
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            eprintln!("⚠ GPU support not compiled. Using CPU mode.");
            init_context(
                "Iron Learn",
                6,
                args.data_file,
                args.lr,
                args.epochs,
                false,
                args.adjust_lr,
                args.internal_layers,
                args.parameters_path,
                args.monitor_interval,
                args.sleep_time,
                args.name,
                args.restore,
                distribution,
                args.mode,
                args.predict_only,
                args.resize,
            );
        }
    } else {
        init_context(
            "Iron Learn",
            5,
            args.data_file,
            args.lr,
            args.epochs,
            false,
            args.adjust_lr,
            args.internal_layers,
            args.parameters_path,
            args.monitor_interval,
            args.sleep_time,
            args.name,
            args.restore,
            distribution,
            args.mode,
            args.predict_only,
            args.resize,
        );
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
    println!("║ Model Path: {}", ctx.weights_path);
    println!("║ Monitor Interval: {}", ctx.monitor_interval);
    println!("║ Intermediate Sleep Time: {} seconds", ctx.sleep_time);
    println!("╚═══════════════════════════════════════╝\n");
}
