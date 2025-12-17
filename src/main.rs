use cublas_sys::*;
use cust::prelude::Module;
use cust::stream::Stream;
use cust::stream::StreamFlags;
use iron_learn::run_neural_net;
use iron_learn::Data;
use iron_learn::Tensor;
use iron_learn::{init_context, run_linear, run_logistic, CpuTensor, GpuTensor, GLOBAL_CONTEXT};
use std::env;
use std::time::Instant;
use std::ptr;

fn init() {
    let args: Vec<String> = env::args().collect();

    let gpu_enabled: bool = if args.len() > 1 {
        args[1] != "cpu"
    } else {
        true
    };

    let learning_rate = args
        .get(2)
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.001);
    let epochs = args
        .get(3)
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(10000);
    let data_path = args
        .get(4)
        .map(|s| s.to_owned())
        .unwrap_or_else(|| "data.json".to_owned());

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
                    data_path,
                    learning_rate,
                    epochs,
                    true,
                    Some(context),
                    Some(module),
                    Some(stream),
                    Some(handle),
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

    if ctx.gpu_enabled {
        println!("Running GPU-based training...\n");
        let _ = run_neural_net::<GpuTensor<f64>>();
        println!("\n✓ All training tasks completed");
    } else {
        println!("Running CPU-based training...\n");
        let _ = run_neural_net::<CpuTensor<f64>>();
        println!("\n✓ All training tasks completed");
    }
}
