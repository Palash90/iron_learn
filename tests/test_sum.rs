#[cfg(test)]
mod sum_tests {
    use iron_learn::CpuTensor;
    use iron_learn::Tensor;

    #[test]
    fn test_sum_basic() {
        let t = CpuTensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let res = t.sum().expect("Sum failed");

        assert_eq!(res.get_shape(), &vec![1]);
        assert_eq!(res.get_data()[0], 10.0);
    }

    #[test]
    fn test_sum_empty() {
        let t: CpuTensor<f32> = CpuTensor::new(vec![0], vec![]).unwrap();

        let res = t.sum().unwrap();

        assert_eq!(res.get_data()[0], 0.0);
    }

    #[test]
    fn test_sum_negative_cancellation() {
        let t = CpuTensor::new(vec![4], vec![100.0, -100.0, 50.5, -50.5]).unwrap();

        let res = t.sum().unwrap();

        assert_eq!(res.get_data()[0], 0.0);
    }

    #[test]
    fn test_sum_large_scale_precision() {
        let size = 10_000;
        let val = 0.1_f32;
        let t = CpuTensor::new(vec![size as u32], vec![val; size]).unwrap();

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
        let t = CpuTensor::new(vec![2], vec![1.0, f32::INFINITY]).unwrap();

        let res = t.sum().unwrap();

        assert!(res.get_data()[0].is_infinite());

        let t_nan = CpuTensor::new(vec![2], vec![1.0, f32::NAN]).unwrap();
        let res_nan = t_nan.sum().unwrap();

        assert!(res_nan.get_data()[0].is_nan());
    }
}
