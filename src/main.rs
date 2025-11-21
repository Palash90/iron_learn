use std::env;

use iron_learn::{init_context, run_ml_cuda, GLOBAL_CONTEXT};

fn init() {
    let args: Vec<String> = env::args().collect();

    if args.len() > 1 && args[1] == "gpu" {
        // Initialize CUDA and create a default context
        let _context = cust::quick_init();

        match _context {
            Err(e) => {
                eprintln!("Failed to initialize CUDA: {}", e);
                init_context("Iron Learn", 1, false, None);
            }
            Ok(ctx) => {
                init_context("Iron Learn", 1, true, Some(ctx));
            }
        };
    } else {
        init_context("Iron Learn", 1, false, None);
    }
}

fn main() {
    init();
    let ctx = GLOBAL_CONTEXT.get().expect("Context not initialized");
    println!("Welcome to {} v{}.", ctx.app_name, ctx.version);
    match ctx.gpu_enabled {
        true => println!("GPU is enabled."),
        false => println!("GPU is not enabled."),
    }

    match run_ml_cuda() {
        Ok(_) => println!("CUDA ML run completed successfully."),
        Err(e) => eprintln!("CUDA ML run failed: {}", e),
    }
}
