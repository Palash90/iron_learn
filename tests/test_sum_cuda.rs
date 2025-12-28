#[cfg(test)]
mod sum_tests {
    use iron_learn::GpuTensor;
    use iron_learn::Tensor;

    use iron_learn::init_context;
    use iron_learn::init_gpu;

    use cublas_sys::*;
    use cust::prelude::Module;
    use cust::stream::Stream;
    use cust::stream::StreamFlags;
    use std::ptr;

    use iron_learn::neural_network::DistributionType;

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
    fn test_sum_basic() {
        init();

        let t: GpuTensor<f32> = GpuTensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let res = t.sum().expect("Sum failed");

        assert_eq!(res.get_shape(), &vec![1]);
        assert_eq!(res.get_data()[0], 10.0);
    }

    #[test]
    fn test_sum_negative_cancellation() {
        init();

        let t = GpuTensor::<f32>::new(vec![4], vec![100.0, -100.0, 50.5, -50.5]).unwrap();

        let res = t.sum().unwrap();

        assert_eq!(res.get_data()[0], 0.0);
    }

    #[test]
    fn test_sum_large_scale_precision() {
        init();

        let size = 10_000;
        let val = 0.1_f32;
        let t = GpuTensor::new(vec![size as u32], vec![val; size]).unwrap();

        let res = t.sum().unwrap();

        let expected = 1_000.0;
        let actual = res.get_data()[0];
        println!("Expected: {}, Actual: {}", expected, actual);

        let unmatched = t.get_data().iter().filter(|i| **i != 0.1).count();
        println!("Not matched {unmatched}");

        assert!(
            (actual - expected).abs() < 1e-1,
            "Precision loss too high: {}",
            actual
        );
    }

    #[test]
    fn test_sum_inf_nan_handling() {
        init();

        let t = GpuTensor::new(vec![2], vec![1.0, f32::INFINITY]).unwrap();

        let res = t.sum().unwrap();

        assert!(res.get_data()[0].is_infinite());

        let t_nan = GpuTensor::new(vec![2], vec![1.0, f32::NAN]).unwrap();
        let res_nan = t_nan.sum().unwrap();

        assert!(res_nan.get_data()[0].is_nan());
    }
}
