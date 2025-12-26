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

    // Helper to simplify tensor creation in tests
    fn new_gpu_tensor(shape: Vec<u32>, data: Vec<f32>) -> GpuTensor<f32> {
        GpuTensor::<f32>::new(shape, data).unwrap()
    }

    #[test]
    fn test_div_large_data() {
        init();

        // Verifies there are no iterator bottlenecks or index out of bounds
        let size = 10_000;
        let data1 = vec![1.0; size];
        let data2 = vec![2.0; size];
        let t1 = new_gpu_tensor(vec![size as u32], data1);
        let t2 = new_gpu_tensor(vec![size as u32], data2);

        let result = t1.div(&t2).unwrap();
        let data = result.get_data();

        println!(
            "Count of first tensor 1s: {}",
            t1.get_data().iter().filter(|x| **x == 1.0).count()
        );
        println!(
            "Count of second tensor 2s: {}",
            t2.get_data().iter().filter(|x| **x == 2.0).count()
        );
        println!(
            "Count of result with 0.5: {}",
            data.iter().filter(|x| **x == 0.5).count()
        );

        assert_eq!(data.len(), size);
        assert!(data.iter().all(|&x| x == 0.5));
    }

    #[test]
    fn test_hadamard_large_data() {
        init();

        // Verifies there are no iterator bottlenecks or index out of bounds
        let size = 10_000;
        let data1 = vec![1.0; size];
        let data2 = vec![2.0; size];
        let t1 = new_gpu_tensor(vec![size as u32], data1);
        let t2 = new_gpu_tensor(vec![size as u32], data2);

        let result = t1.multiply(&t2).unwrap();
        let data = result.get_data();

        println!(
            "Count of first tensor 1s: {}",
            t1.get_data().iter().filter(|x| **x == 1.0).count()
        );
        println!(
            "Count of second tensor 2s: {}",
            t2.get_data().iter().filter(|x| **x == 2.0).count()
        );
        println!(
            "Count of result with 2.0: {}",
            data.iter().filter(|x| **x == 2.0).count()
        );

        assert_eq!(data.len(), size);
        assert!(data.iter().all(|&x| x == 2.0));
    }

    #[test]
    fn test_add_large_data() {
        init();

        // Verifies there are no iterator bottlenecks or index out of bounds
        let size = 10_000_000;
        let data1 = vec![1.0; size];
        let data2 = vec![2.0; size];
        let t1 = new_gpu_tensor(vec![size as u32], data1);
        let t2 = new_gpu_tensor(vec![size as u32], data2);

        let result = t1.add(&t2).unwrap();
        let data = result.get_data();

        println!(
            "Count of first tensor 1s: {}",
            t1.get_data().iter().filter(|x| **x == 1.0).count()
        );
        println!(
            "Count of second tensor 2s: {}",
            t2.get_data().iter().filter(|x| **x == 2.0).count()
        );
        println!(
            "Count of result with 3.0: {}",
            data.iter().filter(|x| **x == 3.0).count()
        );

        assert_eq!(data.len(), size);
        assert!(data.iter().all(|&x| x == 3.0));
    }

    #[test]
    fn test_sub_large_data() {
        init();

        // Verifies there are no iterator bottlenecks or index out of bounds
        let size = 10_000_000;
        let data1 = vec![1.0; size];
        let data2 = vec![2.0; size];
        let t1 = new_gpu_tensor(vec![size as u32], data1);
        let t2 = new_gpu_tensor(vec![size as u32], data2);

        let result = t1.sub(&t2).unwrap();
        let data = result.get_data();

        println!(
            "Count of first tensor 1s: {}",
            t1.get_data().iter().filter(|x| **x == 1.0).count()
        );
        println!(
            "Count of second tensor 2s: {}",
            t2.get_data().iter().filter(|x| **x == 2.0).count()
        );
        println!(
            "Count of result with -1.0: {}",
            data.iter().filter(|x| **x == -1.0).count()
        );

        assert_eq!(data.len(), size);
        assert!(data.iter().all(|&x| x == -1.0));
    }

    #[test]
    fn test_matmul_large_gpu_data() {
        init();

        let rows_a = 9999;
        let inner_k = 999;
        let cols_b = 9999;

        let all_a = 45.0_f32;
        let all_b = 2.0_f32;

        let data1 = vec![all_a; rows_a * inner_k];
        let data2 = vec![all_b; inner_k * cols_b];

        let t1 = new_gpu_tensor(vec![rows_a as u32, inner_k as u32], data1);
        let t2 = new_gpu_tensor(vec![inner_k as u32, cols_b as u32], data2);

        let result = t1.mul(&t2).expect("GPU Matmul failed");
        let result_data = result.get_data();

        assert_eq!(result.get_shape(), &vec![rows_a as u32, cols_b as u32]);
        assert_eq!(result_data.len(), rows_a * cols_b);

        assert!(result_data
            .iter()
            .all(|&x| x == (all_a * all_b * inner_k as f32)));

        println!(
            "Matmul verification successful. All sampled elements equal {}",
            inner_k
        );
    }
}
