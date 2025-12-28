use clap::Parser;
use cublas_sys::*;
use cust::prelude::Module;
use cust::stream::Stream;
use cust::stream::StreamFlags;
use iron_learn::init_gpu;
use iron_learn::neural_network::DistributionType;
use iron_learn::run_neural_net;
use iron_learn::{init_context, CpuTensor, GpuTensor, GLOBAL_CONTEXT};
use std::ptr;

#[derive(Parser)]
#[command(name = "Iron Learn")]
#[command(name = "A Rust Machine Learning Library")]
struct Args {
    #[arg(long, short, default_value = "neural-net")]
    name: String,

    #[arg(long, short, default_value = "false")]
    cpu: bool,

    #[arg(long, short, default_value = "false")]
    restore: bool,

    #[arg(long, short, default_value = "0.001")]
    lr: f64,

    #[arg(long, short, default_value = "10000")]
    epochs: u32,

    #[arg(long, short, default_value = "data/linear.json")]
    data_file: String,

    #[arg(long, short, default_value = "false")]
    adjust_lr: bool,

    #[arg(long, short, default_value = "4")]
    internal_layers: u32,

    #[arg(long, short, default_value = "1000")]
    monitor_interval: usize,

    #[arg(long, short, default_value = "30")]
    sleep_time: u64,

    #[arg(long, short, default_value = "model.json")]
    parameters_path: String,

    #[arg(long, short='D', default_value = "Simple")]
    distribution: String,
}

pub fn init() {
    let args = Args::parse();
    let gpu_enabled = !args.cpu;

    let distribution = match args.distribution.as_str().to_uppercase().as_str() {
        "NORMAL" => DistributionType::Normal,
        "XAVIER" => DistributionType::Xavier,
        "SIMPLE" => DistributionType::Simple,
        _ => DistributionType::Normal,
    };

    if gpu_enabled {
        match cust::quick_init() {
            Ok(context) => {
                println!("✓ GPU initialization successful");

                let mut handle: cublasHandle_t = ptr::null_mut();
                unsafe {
                    let status = cublasCreate_v2(&mut handle);
                    if status != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                        eprintln!("Failed to create cuBLAS handle");
                        return;
                    }
                };

                println!("✓ CUBLAS initialization successful");

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
                );

                init_gpu(Some(context), Some(module), Some(stream), Some(handle));
            }
            Err(e) => {
                eprintln!("⚠ GPU initialization failed: {}. Using CPU mode.", e);
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
                );
            }
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
        );
    }
}

fn greet(ctx: &iron_learn::AppContext) {
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

fn main() {
    init();

    let ctx = GLOBAL_CONTEXT.get().expect("Context not initialized");
    greet(ctx);

    if ctx.gpu_enabled {
        println!("Running GPU-based training...\n");
        let _ = run_neural_net::<GpuTensor<f32>>();
        println!("\n✓ All training tasks completed");
    } else {
        println!("Running CPU-based training...\n");
        let _ = run_neural_net::<CpuTensor<f32>>();
        println!("\n✓ All training tasks completed");
    }
}
