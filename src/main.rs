use cust::prelude::Module;
use cust::stream::Stream;
use cust::stream::StreamFlags;
use iron_learn::gpu_ml;
use iron_learn::run_neural_net;
use iron_learn::Tensor;
use iron_learn::{init_context, run_linear, run_logistic, CpuTensor, GpuTensor, GLOBAL_CONTEXT};
use std::env;
use std::time::Instant;

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
        .unwrap_or(0.01);
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
                eprintln!("✓ GPU initialization successful");

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

        let now = Instant::now();
        let _ = run_linear::<GpuTensor<f64>>();
        let elapsed = now.elapsed();
        println!("Linear Regression completed in {:.4?}", elapsed);

        let now = Instant::now();
        let _ = run_logistic::<GpuTensor<f64>>();
        let elapsed = now.elapsed();
        println!("Logistic Regression completed in {:.4?}", elapsed);

        let now = Instant::now();
        let _ = gpu_ml::run_linear_cuda();
        let elapsed = now.elapsed();
        println!("Old linear Regression completed in {:.4?}", elapsed);

        let now = Instant::now();
        let _ = gpu_ml::run_logistics_cuda();
        let elapsed = now.elapsed();
        println!("Old logistic Regression completed in {:.4?}", elapsed);

        let now = Instant::now();
        gpu_ml::run_neural_network_cuda();
        let elapsed = now.elapsed();
        println!("Neural Net completed in {:.4?}", elapsed);

        let now = Instant::now();
        gpu_ml::examples::example_3layer_network();
        let elapsed = now.elapsed();
        println!("Neural Net completed in {:.4?}", elapsed);

        println!("\n✓ All training tasks completed");
    } else {
        println!("Running CPU-based training...\n");
        run_neural_net::<CpuTensor<f64>>();
        run_linear::<CpuTensor<f64>>();
        run_logistic::<CpuTensor<f64>>();
        println!("\n✓ All training tasks completed");
    }
}
