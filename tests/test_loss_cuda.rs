#[cfg(test)]
mod loss_tests {
    use iron_learn::neural_network::loss_functions::{
        BinaryCrossEntropy, LossFunction, MeanSquaredErrorLoss,
    };
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
    fn test_mse_loss_and_prime_gpu() {
        init();

        let y_true = GpuTensor::new(vec![2], vec![1.0_f32, 2.0_f32]).unwrap();
        let y_pred = GpuTensor::new(vec![2], vec![1.5_f32, 1.5_f32]).unwrap();

        let mse = MeanSquaredErrorLoss;

        let res = mse.loss(&y_true, &y_pred).expect("MSE loss failed");
        assert_eq!(res.get_shape(), &vec![1]);
        assert!((res.get_data()[0] - 0.25).abs() < 1e-6);

        let grad = mse
            .loss_prime(&y_true, &y_pred)
            .expect("MSE loss_prime failed");
        let g = grad.get_data();
        assert!((g[0] - 0.5).abs() < 1e-6);
        assert!((g[1] + 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_bce_loss_and_prime_gpu() {
        init();

        let y_true = GpuTensor::new(vec![2], vec![1.0_f32, 0.0_f32]).unwrap();
        let y_pred = GpuTensor::new(vec![2], vec![0.9_f32, 0.1_f32]).unwrap();

        let bce = BinaryCrossEntropy;

        let res = bce.loss(&y_true, &y_pred).expect("BCE loss failed");
        let val = res.get_data()[0];
        let expected = -2.0_f32 * (0.9_f32.ln());
        assert!(
            (val - expected).abs() < 1e-6,
            "bce loss mismatch: {} vs {}",
            val,
            expected
        );

        let grad = bce
            .loss_prime(&y_true, &y_pred)
            .expect("BCE loss_prime failed");
        let g = grad.get_data();
        assert!((g[0] + 0.55555556).abs() < 1e-6);
        assert!((g[1] - 0.55555556).abs() < 1e-6);
    }
}
