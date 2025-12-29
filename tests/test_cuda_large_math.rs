#![cfg(feature = "cuda")]

#[cfg(test)]
mod tensor_math_large_tests {
    use iron_learn::tensor::math::TensorMath;
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

    // Using the size that resulted in ~1 billion elements from your previous run
    const LARGE_M: u32 = 1000;
    const LARGE_N: u32 = 1000;
    const TOTAL_ELEMENTS: usize = (LARGE_M as usize) * (LARGE_N as usize);

    // Helper to simplify tensor creation in tests
    fn new_gpu_tensor(shape: Vec<u32>, data: Vec<f32>) -> GpuTensor<f32> {
        GpuTensor::<f32>::new(shape, data).unwrap()
    }

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
    fn test_large_exp() {
        init();
        let t1 = new_gpu_tensor(vec![LARGE_M, LARGE_N], vec![0.0f32; TOTAL_ELEMENTS]);

        // TEST EXP
        let res_exp = t1.exp().expect("Exp failed");
        let data_exp = res_exp.get_data();
        assert_eq!(data_exp[0], 1.0);
        assert_eq!(data_exp[TOTAL_ELEMENTS - 1], 1.0);
    }

    #[test]
    fn test_large_sigmoid() {
        init();
        let t1 = new_gpu_tensor(vec![LARGE_M, LARGE_N], vec![0.0f32; TOTAL_ELEMENTS]);

        // TEST SIGMOID
        let res_sig = t1.sigmoid().expect("Sigmoid failed");
        let data_sig = res_sig.get_data();
        assert_eq!(data_sig[0], 0.5);
        assert_eq!(data_sig[TOTAL_ELEMENTS / 2], 0.5);
    }

    #[test]
    fn test_large_log_stability() {
        init();
        // GIVEN: e^1 (approx 2.71828)
        let e = std::f32::consts::E;
        let t1 = new_gpu_tensor(vec![LARGE_M, LARGE_N], vec![e; TOTAL_ELEMENTS]);

        // WHEN: ln(e)
        let result = t1.ln().expect("Ln failed");
        let data = result.get_data();

        // THEN: should be approx 1.0
        let diff = (data[0] - 1.0).abs();
        assert!(diff < 1e-5);
    }

    #[test]
    fn test_large_tanh_range() {
        init();
        // GIVEN: Large positive and negative numbers
        let mut input_data = vec![0.0f32; TOTAL_ELEMENTS];
        input_data[0] = 100.0;
        input_data[TOTAL_ELEMENTS - 1] = -100.0;

        let t1 = new_gpu_tensor(vec![LARGE_M, LARGE_N], input_data);

        // WHEN: tanh
        let result = t1.tanh().expect("Tanh failed");
        let data = result.get_data();

        // THEN: tanh(100) -> 1.0, tanh(-100) -> -1.0
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[TOTAL_ELEMENTS - 1] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_trig_functions() {
        init();
        // GIVEN: PI / 2
        let pi_2 = std::f32::consts::FRAC_PI_2;
        let t1 = new_gpu_tensor(vec![100, 100], vec![pi_2; 10000]);

        // sin(pi/2) = 1.0; cos(pi/2) = 0.0
        let s = t1.sin().unwrap().get_data();
        let c = t1.cos().unwrap().get_data();

        assert!((s[0] - 1.0).abs() < 1e-6);
        assert!(c[0].abs() < 1e-6);
    }

    #[test]
    fn test_large_log_and_ln() {
        init();
        let size = 10_000; // Testing at a manageable but large scale
        let total = size * size;

        // Test ln (Base e): ln(e^2) should be 2.0
        let e2 = std::f32::consts::E.powi(2);
        let t_ln = new_gpu_tensor(vec![size as u32, size as u32], vec![e2; total]);
        let res_ln = t_ln.ln().expect("Ln failed");
        assert!((res_ln.get_data()[0] - 2.0).abs() < 1e-5);

        // Test log (Base 10): log10(1000.0) should be 3.0
        let t_log = new_gpu_tensor(vec![size as u32, size as u32], vec![1000.0; total]);
        let res_log = t_log.log().expect("Log failed");
        assert!((res_log.get_data()[0] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_large_tan_and_periodicity() {
        init();
        let total = 10_000;

        // tan(pi/4) = 1.0
        let pi_4 = std::f32::consts::FRAC_PI_4;
        let t1 = new_gpu_tensor(vec![total as u32], vec![pi_4; total]);

        let res = t1.tan().expect("Tan failed");
        let data = res.get_data();

        // Check first and last to ensure the whole buffer was processed
        assert!((data[0] - 1.0).abs() < 1e-5);
        assert!((data[total - 1] - 1.0).abs() < 1e-5);
    }
}
