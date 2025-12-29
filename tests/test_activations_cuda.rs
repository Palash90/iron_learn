#[cfg(test)]
mod activations_tests {
    use iron_learn::neural_network::{cos, sigmoid, sigmoid_prime, sin, tanh, tanh_prime};
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
    fn test_sigmoid_and_prime_gpu() {
        init();

        let vals = vec![0.0_f32, 1.0_f32, -1.0_f32, 2.0_f32];
        let x = GpuTensor::new(vec![4], vals.clone()).unwrap();

        let s = sigmoid(&x).expect("sigmoid failed");
        let sdata = s.get_data();

        let eps = 1e-6_f32;
        for (i, v) in vals.iter().enumerate() {
            let expected = 1.0_f32 / (1.0_f32 + (-v).exp());
            assert!(
                (sdata[i] - expected).abs() < eps,
                "sigmoid mismatch at {}",
                i
            );
        }

        let sp = sigmoid_prime(&s).expect("sigmoid_prime failed");
        let spd = sp.get_data();
        for (i, sd) in sdata.iter().enumerate() {
            let expected = sd * (1.0_f32 - sd);
            assert!(
                (spd[i] - expected).abs() < eps,
                "sigmoid_prime mismatch at {}",
                i
            );
        }
    }

    #[test]
    fn test_tanh_and_prime_gpu() {
        init();

        let vals = vec![0.0_f32, 1.0_f32, -1.0_f32, 0.5_f32];
        let x = GpuTensor::new(vec![4], vals.clone()).unwrap();

        let t = tanh(&x).expect("tanh failed");
        let tdata = t.get_data();

        let eps = 1e-6_f32;
        for (i, v) in vals.iter().enumerate() {
            let expected = v.tanh();
            assert!((tdata[i] - expected).abs() < eps, "tanh mismatch at {}", i);
        }

        let tp = tanh_prime(&t).expect("tanh_prime failed");
        let tpd = tp.get_data();
        for (i, td) in tdata.iter().enumerate() {
            let expected = 1.0_f32 - td * td;
            assert!(
                (tpd[i] - expected).abs() < eps,
                "tanh_prime mismatch at {}",
                i
            );
        }
    }

    #[test]
    fn test_sin_and_cos_gpu() {
        init();

        let vals = vec![0.0_f32, std::f32::consts::FRAC_PI_2, -1.0_f32, 2.0_f32];
        let x = GpuTensor::new(vec![4], vals.clone()).unwrap();

        let s = sin(&x).expect("sin failed");
        let sdata = s.get_data();
        let c = cos(&x).expect("cos failed");
        let cdata = c.get_data();

        let eps = 1e-6_f32;
        for (i, v) in vals.iter().enumerate() {
            let expected_s = v.sin();
            let expected_c = v.cos();
            assert!((sdata[i] - expected_s).abs() < eps, "sin mismatch at {}", i);
            assert!((cdata[i] - expected_c).abs() < eps, "cos mismatch at {}", i);
        }
    }
}
