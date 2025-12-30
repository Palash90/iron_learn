use iron_learn::init::init_runtime;
use iron_learn::run_neural_net;

use iron_learn::CpuTensor;

#[cfg(feature = "cuda")]
use iron_learn::GpuTensor;

fn main() {
    init_runtime();
    #[cfg(feature = "cuda")]
    {
        if ctx.gpu_enabled {
            println!("Running GPU-based training...\n");
            let _ = run_neural_net::<GpuTensor<f32>>();
            println!("\n✓ All training tasks completed");
            return;
        }
    }

    println!("Running CPU-based training...\n");
    let _ = run_neural_net::<CpuTensor<f32>>();
    println!("\n✓ All training tasks completed");
}
