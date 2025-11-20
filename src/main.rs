use iron_learn::*;
use std::env;
use std::fs;

pub fn run_ml() {
    let l = 0.001;

    let e = 5000;

    let contents = fs::read_to_string("data.json").expect("Should have been able to read the file");

    // Parse once and destructure into `linear` and `logistic` datasets
    let data: Data = serde_json::from_str(&contents).unwrap();
    let Data {
        linear: xy,
        logistic,
    } = data;

    // Run linear regression using the extracted function
    run_linear(&xy, l, e);

    // Call the extracted function to run logistic regression using the `logistic` dataset
    run_logistic(&logistic, l, e);
}

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

    let A = Tensor::new(
        vec![4, 4],
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
    )
    .unwrap();
    let B = Tensor::new(
        vec![4, 4],
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
    )
    .unwrap();
    println!("Tensor A:\n{:?}", A);
    println!("Tensor B:\n{:?}", B);
    let C = (A * B).unwrap();

    println!("Tensor C (A + B):\n{:?}", C);
}
