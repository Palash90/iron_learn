#[cfg(test)]
mod loss_tests {
    use iron_learn::neural_network::loss_functions::{
        BinaryCrossEntropy, LossFunction, MeanSquaredErrorLoss,
    };
    use iron_learn::CpuTensor;
    use iron_learn::Tensor;

    #[test]
    fn test_mse_loss_and_prime() {
        // actual: [1,2], predicted: [1.5,1.5]
        let y_true = CpuTensor::new(vec![2], vec![1.0_f32, 2.0_f32]).unwrap();
        let y_pred = CpuTensor::new(vec![2], vec![1.5_f32, 1.5_f32]).unwrap();

        let mse = MeanSquaredErrorLoss;

        // Loss: mean((pred - actual)^2) = mean([0.25,0.25]) = 0.25
        let res = mse.loss(&y_true, &y_pred).expect("MSE loss failed");
        assert_eq!(res.get_shape(), &vec![1]);
        let val = res.get_data()[0];
        assert!((val - 0.25).abs() < 1e-6, "mse loss mismatch: {}", val);

        // Loss prime: (pred - actual) * (2/n) => n=2 => factor=1
        let grad = mse
            .loss_prime(&y_true, &y_pred)
            .expect("MSE loss_prime failed");

        let gdata = grad.get_data();
        assert!((gdata[0] - 0.5).abs() < 1e-6);
        assert!((gdata[1] + 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_binary_cross_entropy_loss_and_prime() {
        // y_true: [1,0], y_pred: [0.9, 0.1]
        let y_true = CpuTensor::new(vec![2], vec![1.0_f32, 0.0_f32]).unwrap();
        let y_pred = CpuTensor::new(vec![2], vec![0.9_f32, 0.1_f32]).unwrap();

        let bce = BinaryCrossEntropy;

        // Loss (implemented as sum(-[y ln(y_hat) + (1-y) ln(1-y_hat)]))
        // For these values: -2 * ln(0.9)
        let res = bce.loss(&y_true, &y_pred).expect("BCE loss failed");
        let val = res.get_data()[0];
        let expected = -2.0_f32 * (0.9_f32.ln());
        assert!(
            (val - expected).abs() < 1e-6,
            "bce loss mismatch: {} vs {}",
            val,
            expected
        );

        // Loss prime: ((p - y) / (p*(1-p))) / size
        let grad = bce
            .loss_prime(&y_true, &y_pred)
            .expect("BCE loss_prime failed");

        let g = grad.get_data();
        // As calculated: numerator [-0.1, 0.1], denom [0.09,0.09] => [-1.111111,1.111111]/2
        assert!((g[0] + 0.55555556).abs() < 1e-6, "grad0 {}", g[0]);
        assert!((g[1] - 0.55555556).abs() < 1e-6, "grad1 {}", g[1]);
    }
}
