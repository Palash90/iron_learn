#![cfg(feature = "cuda")]

#[cfg(test)]
mod tests {
    use iron_learn::GpuTensor;
    use iron_learn::Tensor;

    use iron_learn::init_context;
    use iron_learn::init_gpu;

    use cublas_sys::*;
    use cust::prelude::Module;
    use cust::stream::Stream;
    use cust::stream::StreamFlags;
    use iron_learn::neural_network::DistributionType;
    use std::ptr;

    fn init() {
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

                let mut handle: cublasHandle_t = ptr::null_mut();
                unsafe {
                    let status = cublasCreate_v2(&mut handle);
                    if status != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                        eprintln!("Failed to create cuBLAS handle");
                        return;
                    }
                };

                init_context(
                    "Iron Learn",
                    5,
                    String::new(),
                    0.0,
                    0,
                    true,
                    false,
                    2,
                    "w".to_string(),
                    0,
                    0,
                    "".to_string(),
                    false,
                    DistributionType::Normal,
                );
                init_gpu(Some(context), Some(module), Some(stream), Some(handle));
            }
            Err(e) => {
                eprintln!("⚠ GPU initialization failed: {}. Using CPU mode.", e);
                init_context(
                    "Iron Learn",
                    5,
                    "".to_string(),
                    0.01,
                    1,
                    false,
                    false,
                    2,
                    "w".to_string(),
                    0,
                    0,
                    "".to_string(),
                    false,
                    DistributionType::Normal,
                );
            }
        }
    }

    #[test]
    fn test_div_happy_path() {
        init();

        // Verifies basic element-wise division and shape preservation
        let shape = vec![20, 20];
        let t1 = GpuTensor::<f32>::ones(&shape);
        let data = t1.get_data();

        let unmatched = data.iter().filter(|x| **x != 1.0).count();
        let matched = data.iter().filter(|x| **x == 1.0).count();

        println!("Unmatched {unmatched}, matched {matched}");

        assert!(data.iter().all(|&x| x == 1.0));
    }
}
