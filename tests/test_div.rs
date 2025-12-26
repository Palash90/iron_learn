#[cfg(test)]
mod tests {
    use super::*;
    use iron_learn::CpuTensor;
    use iron_learn::Numeric;
    use iron_learn::Tensor;

    // Helper to simplify tensor creation in tests
    fn new_cpu_tensor<T: Numeric>(shape: Vec<u32>, data: Vec<T>) -> CpuTensor<T> {
        CpuTensor::new(shape, data).unwrap()
    }

    #[test]
    fn test_div_happy_path() {
        // Verifies basic element-wise division and shape preservation
        let shape = vec![2, 2];
        let t1 = new_cpu_tensor(shape.clone(), vec![10.0, 20.0, 30.0, 40.0]);
        let t2 = new_cpu_tensor(shape.clone(), vec![2.0, 4.0, 5.0, 8.0]);

        let result = t1.div(&t2).expect("Division should succeed");

        assert_eq!(result.get_data(), vec![5.0, 5.0, 6.0, 5.0]);
        assert_eq!(result.get_shape(), &shape);
    }

    #[test]
    fn test_div_shape_mismatch_error() {
        // Verifies that different ranks or dimensions trigger the error
        let t1 = new_cpu_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let t2 = new_cpu_tensor(vec![4], vec![1.0, 2.0, 3.0, 4.0]);

        let result = t1.div(&t2);

        assert!(result.is_err());
        let err_msg = result.unwrap_err();
        assert!(err_msg.contains("ShapeMismatch"));
        assert!(err_msg.contains("[2, 2]")); // Check if error reports shapes correctly
    }

    #[test]
    fn test_div_scalar_like_tensor() {
        // Verifies a 1x1 tensor (common edge case in linear algebra)
        let t1 = new_cpu_tensor(vec![1], vec![100.0]);
        let t2 = new_cpu_tensor(vec![1], vec![10.0]);

        let result = t1.div(&t2).unwrap();
        assert_eq!(result.get_data(), vec![10.0]);
    }

    #[test]
    fn test_div_precision_and_nan() {
        // Verifies behavior with floating point limits
        // Note: Assuming T is f32/f64 for this specific test
        let t1 = new_cpu_tensor(vec![2], vec![0.0, f32::INFINITY]);
        let t2 = new_cpu_tensor(vec![2], vec![0.0, f32::INFINITY]);

        let result = t1.div(&t2).unwrap();

        // 0/0 is NaN, Inf/Inf is NaN
        assert!(result.get_data()[0].is_nan());
        assert!(result.get_data()[1].is_nan());
    }

    #[test]
    fn test_div_by_zero() {
        // Verifies behavior with floating point limits
        // Note: Assuming T is f32/f64 for this specific test
        let t1 = new_cpu_tensor(vec![2], vec![1.0, 1.0]);
        let t2 = new_cpu_tensor(vec![2], vec![0.0, f32::INFINITY]);

        let result = t1.div(&t2).unwrap();

        // 0/0 is NaN, Inf/Inf is NaN
        assert!(result.get_data()[0].is_infinite());
        assert!(result.get_data()[1] == 0.0);
    }

    #[test]
    fn test_div_large_data() {
        // Verifies there are no iterator bottlenecks or index out of bounds
        let size = 10_000;
        let data1 = vec![1.0; size];
        let data2 = vec![2.0; size];
        let t1 = new_cpu_tensor(vec![size as u32], data1);
        let t2 = new_cpu_tensor(vec![size as u32], data2);

        let result = t1.div(&t2).unwrap();
        assert_eq!(result.get_data().len(), size);
        assert!(result.get_data().iter().all(|&x| x == 0.5));
    }
}
