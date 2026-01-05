use iron_learn::examples::init::ExampleMode;
use iron_learn::examples::trigram::run_trigram_generator;
use iron_learn::examples::{run_bigram_generator, run_linear, run_logistic, run_neural_net};

use iron_learn::numeric::FloatingPoint;
use iron_learn::tensor::math::TensorMath;
use iron_learn::{CpuTensor, Tensor};

#[cfg(feature = "cuda")]
use iron_learn::GpuTensor;

fn run_ml<T, D>(mode: ExampleMode)
where
    T: Tensor<D> + TensorMath<D, MathOutput = T> + 'static,
    D: FloatingPoint + 'static,
{
    match mode {
        ExampleMode::Linear => match run_linear::<T, D>() {
            Ok(_) => (),
            Err(e) => eprintln!("Error: {}", e),
        },
        ExampleMode::Logistic => match run_logistic::<T, D>() {
            Ok(_) => (),
            Err(e) => eprintln!("Error: {}", e),
        },
        ExampleMode::Bigram => match run_bigram_generator::<T, D>() {
            Ok(_) => (),
            Err(e) => eprintln!("Error: {}", e),
        },
        ExampleMode::Trigram => match run_trigram_generator::<T, D>() {
            Ok(_) => (),
            Err(e) => eprintln!("Error: {}", e),
        },
        _ => match run_neural_net::<T, D>() {
            Ok(_) => (),
            Err(e) => eprintln!("Error: {}", e),
        },
    };
}

fn main() {
    let ctx = iron_learn::examples::init::init_runtime();
    type DataType = f32;

    #[cfg(feature = "cuda")]
    {
        if ctx.gpu_enabled {
            println!("Running GPU-based training...\n");
            let _ = run_ml::<GpuTensor<DataType>, DataType>(ctx.example_mode);
            println!("\n✓ All training tasks completed");
            return;
        }
    }

    println!("Running CPU-based training...\n");
    let _ = run_ml::<CpuTensor<DataType>, DataType>(ctx.example_mode);
    println!("\n✓ All training tasks completed");
}
