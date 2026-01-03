use iron_learn::examples::{run_linear, run_logistic, run_neural_net};

use iron_learn::numeric::FloatingPoint;
use iron_learn::tensor::math::TensorMath;
use iron_learn::{CpuTensor, Tensor};

#[cfg(feature = "cuda")]
use iron_learn::GpuTensor;

fn run_ml<T, D>()
where
    T: Tensor<D> + TensorMath<D, MathOutput = T> + 'static,
    D: FloatingPoint + 'static,
{
    let _ = run_linear::<T, D>();
    let _ = run_logistic::<T, D>();
    let _ = run_neural_net::<T, D>();
}

fn main() {
    let _ctx = iron_learn::examples::init::init_runtime();

    #[cfg(feature = "cuda")]
    {
        if _ctx.gpu_enabled {
            println!("Running GPU-based training...\n");
            let _ = run_ml::<GpuTensor<f32>, f32>();
            println!("\n✓ All training tasks completed");
            return;
        }
    }

    println!("Running CPU-based training...\n");
    let _ = run_ml::<CpuTensor<f32>, f32>();
    println!("\n✓ All training tasks completed");
}
