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
                );
            }
        }
    }

    #[test]
    fn test_cuda_clip_basic_bounds() {
        init();

        // Setup data with values below, inside, and above range
        let input = vec![-10.0, 0.0, 5.0, 10.0, 20.0];
        let min = 0.0;
        let max = 10.0;

        // Expected: [-10 -> 0], [0 -> 0], [5 -> 5], [10 -> 10], [20 -> 10]
        let expected = vec![0.0, 0.0, 5.0, 10.0, 10.0];

        let tensor = GpuTensor::<f32>::new(vec![5], input).unwrap();
        let clipped = tensor.clip(min, max).unwrap();

        assert_eq!(clipped.get_data(), expected);
    }

    #[test]
    fn test_cuda_clip_all_below() {
        init();

        let input = vec![-5.0, -2.0, -1.0];
        let min = 0.0;
        let max = 10.0;
        let expected = vec![0.0, 0.0, 0.0];

        let tensor = GpuTensor::<f32>::new(vec![3], input).unwrap();
        let clipped = tensor.clip(min, max).unwrap();
        assert_eq!(clipped.get_data(), expected);
    }

    #[test]
    fn test_cuda_clip_all_above() {
        init();

        let input = vec![15.0, 20.0, 100.0];
        let min = 0.0;
        let max = 10.0;
        let expected = vec![10.0, 10.0, 10.0];

        let tensor = GpuTensor::<f32>::new(vec![3], input).unwrap();
        let clipped = tensor.clip(min, max).unwrap();
        assert_eq!(clipped.get_data(), expected);
    }

    #[test]
    fn test_cuda_clip_identical_bounds() {
        init();

        // If min == max, everything should become that value
        let input = vec![1.0, 2.0, 3.0];
        let min = 5.0;
        let max = 5.0;
        let expected = vec![5.0, 5.0, 5.0];

        let tensor = GpuTensor::<f32>::new(vec![3], input).unwrap();
        let clipped = tensor.clip(min, max).unwrap();
        assert_eq!(clipped.get_data(), expected);
    }

    #[test]
    fn test_cuda_clip_with_nans() {
        init();

        // Note: Floating point NaNs usually fail PartialOrd comparisons
        // Depending on your requirements, you might want to handle this explicitly
        let input = vec![f32::NAN, 5.0];
        let min = 0.0;
        let max = 10.0;

        let tensor = GpuTensor::<f32>::new(vec![2], input).unwrap();
        let clipped = tensor.clip(min, max).unwrap();

        // In the standard if/else logic, NaN < min is false and NaN > max is false
        // So NaN usually stays NaN.
        assert!(clipped.get_data()[0].is_nan());
        assert_eq!(clipped.get_data()[1], 5.0);
    }
}
