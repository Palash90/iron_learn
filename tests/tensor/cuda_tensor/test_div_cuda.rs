#![cfg(feature = "cuda")]

#[cfg(test)]
mod tests {
    use iron_learn::GpuTensor;
    use iron_learn::Tensor;

    use iron_learn::init_gpu;

    // Helper to simplify tensor creation in tests
    fn new_gpu_tensor(shape: Vec<u32>, data: Vec<f32>) -> GpuTensor<f32> {
        GpuTensor::<f32>::new(shape, data).unwrap()
    }

    #[test]
    fn test_div_happy_path() {
        let _ = init_gpu();

        // Verifies basic element-wise division and shape preservation
        let shape = vec![2, 2];
        let t1 = new_gpu_tensor(shape.clone(), vec![10.0, 20.0, 30.0, 40.0]);
        let t2 = new_gpu_tensor(shape.clone(), vec![2.0, 4.0, 5.0, 8.0]);

        let result = t1.div(&t2).expect("Division should succeed");

        assert_eq!(result.get_data(), vec![5.0, 5.0, 6.0, 5.0]);
        assert_eq!(result.get_shape(), &shape);
    }

    #[test]
    fn test_div_shape_mismatch_error() {
        let _ = init_gpu();

        // Verifies that different ranks or dimensions trigger the error
        let t1 = new_gpu_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let t2 = new_gpu_tensor(vec![4], vec![1.0, 2.0, 3.0, 4.0]);

        let result = t1.div(&t2);

        assert!(result.is_err());
        let err_msg = result.unwrap_err();
        assert!(err_msg.contains("ShapeMismatch"));
        assert!(err_msg.contains("[2, 2]")); // Check if error reports shapes correctly
    }

    #[test]
    fn test_div_scalar_like_tensor() {
        let _ = init_gpu();

        // Verifies a 1x1 tensor (common edge case in linear algebra)
        let t1 = new_gpu_tensor(vec![1], vec![100.0]);
        let t2 = new_gpu_tensor(vec![1], vec![10.0]);

        let result = t1.div(&t2).unwrap();
        assert_eq!(result.get_data(), vec![10.0]);
    }

    #[test]
    fn test_cuda_div_precision_and_nan() {
        let _ = init_gpu();

        // Verifies behavior with floating point limits
        // Note: Assuming T is f32/f64 for this specific test
        let t1 = new_gpu_tensor(vec![2], vec![0.0, f32::INFINITY]);
        let t2 = new_gpu_tensor(vec![2], vec![0.0, f32::INFINITY]);

        let result = t1.div(&t2).unwrap();

        let data = result.get_data();
        println!("Result: {:?}", data);

        // 0/0 is NaN, Inf/Inf is NaN
        assert!(data[0].is_nan());
        assert!(data[1].is_nan());
    }

    #[test]
    fn test_cuda_div_by_zero() {
        let _ = init_gpu();

        let t1 = new_gpu_tensor(vec![2], vec![1.0, 1.0]);
        let t2 = new_gpu_tensor(vec![2], vec![0.0, f32::INFINITY]);

        let result = t1.div(&t2).unwrap();

        let data = result.get_data();
        println!("Result: {:?}", data);

        assert!(data[0].is_infinite());
        assert!(data[1] == 0.0);
    }
}
