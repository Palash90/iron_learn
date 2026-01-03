#![cfg(feature = "cuda")]

#[cfg(test)]
mod loss_tests {
    use iron_learn::nn::loss_functions::{BinaryCrossEntropy, LossFunction, MeanSquaredErrorLoss};
    use iron_learn::GpuTensor;
    use iron_learn::Tensor;

    use iron_learn::init_gpu;

    #[test]
    fn test_mse_loss_and_prime_gpu() {
        let _ = init_gpu();

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
        let _ = init_gpu();

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
