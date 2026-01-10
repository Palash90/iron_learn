#[cfg(test)]
mod loss_tests {

    use iron_learn::GpuTensor;
    use iron_learn::Tensor;

    use iron_learn::init_gpu;
    use iron_learn::nn::loss_functions::bce;
    use iron_learn::nn::loss_functions::bce_prime;
    use iron_learn::nn::loss_functions::cce;
    use iron_learn::nn::loss_functions::cce_prime;
    use iron_learn::nn::loss_functions::mse;
    use iron_learn::nn::loss_functions::mse_prime;

    #[test]
    fn test_mse_loss_and_prime_gpu() {
        let _ = init_gpu();

        let y_true = GpuTensor::new(vec![2], vec![1.0_f32, 2.0_f32]).unwrap();
        let y_pred = GpuTensor::new(vec![2], vec![1.5_f32, 1.5_f32]).unwrap();

        let res = mse(&y_true, &y_pred).expect("MSE loss failed");
        assert_eq!(res.get_shape(), &vec![1]);
        assert!((res.get_data()[0] - 0.25).abs() < 1e-6);

        let grad = mse_prime(&y_true, &y_pred).expect("MSE loss_prime failed");
        let g = grad.get_data();
        assert!((g[0] - 0.5).abs() < 1e-6);
        assert!((g[1] + 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_bce_loss_and_prime_gpu() {
        let _ = init_gpu();

        let y_true = GpuTensor::new(vec![2], vec![1.0_f32, 0.0_f32]).unwrap();
        let y_pred = GpuTensor::new(vec![2], vec![0.9_f32, 0.1_f32]).unwrap();

        let res = bce(&y_true, &y_pred).expect("BCE loss failed");
        let val = res.get_data()[0];
        let expected = -2.0_f32 * (0.9_f32.ln());
        assert!(
            (val - expected).abs() < 1e-6,
            "bce loss mismatch: {} vs {}",
            val,
            expected
        );

        let grad = bce_prime(&y_true, &y_pred).expect("BCE loss_prime failed");
        let g = grad.get_data();
        assert!((g[0] + 0.555_555_6).abs() < 1e-6);
        assert!((g[1] - 0.555_555_6).abs() < 1e-6);
    }

    #[test]
    fn test_cce_loss_and_prime_gpu() {
        let _ = init_gpu();

        let y_true = GpuTensor::new(vec![2, 3], vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).unwrap();

        let y_pred = GpuTensor::new(vec![2, 3], vec![0.7, 0.2, 0.1, 0.1, 0.3, 0.6]).unwrap();

        let res = cce(&y_true, &y_pred).expect("CCE loss failed");
        let val = res.get_data()[0];

        let expected_loss = -(0.7_f32.ln() + 0.6_f32.ln());

        assert!(
            (val - expected_loss).abs() < 1e-6,
            "CCE loss mismatch: {} vs {}",
            val,
            expected_loss
        );

        let grad = cce_prime(&y_true, &y_pred).expect("CCE loss_prime failed");
        let g = grad.get_data();

        let expected_grads = [-0.15, 0.1, 0.05, 0.05, 0.15, -0.2];

        for i in 0..g.len() {
            assert!(
                (g[i] - expected_grads[i]).abs() < 1e-6,
                "CCE gradient mismatch at index {}: {} vs {}",
                i,
                g[i],
                expected_grads[i]
            );
        }
    }

    #[test]
    fn test_cce_numerical_stability() {
        let _ = init_gpu();

        let y_true = GpuTensor::new(vec![1, 2], vec![1.0, 0.0]).unwrap();
        let y_pred = GpuTensor::new(vec![1, 2], vec![0.0, 1.0]).unwrap(); // Total wrong prediction

        let res = cce(&y_true, &y_pred).expect("CCE stability test failed");
        let val: f32 = res.get_data()[0];

        assert!(
            !val.is_nan(),
            "CCE loss resulted in NaN; check clipping logic"
        );
        assert!(
            val > 0.0,
            "CCE loss should be a large positive number for total misses"
        );
    }
}
