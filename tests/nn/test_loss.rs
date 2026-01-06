#[cfg(test)]
mod loss_tests {
    use iron_learn::nn::loss_functions::{
        BinaryCrossEntropy, CategoricalCrossEntropy, LossFunction, MeanSquaredErrorLoss,
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

    #[test]
    fn test_cce_loss_and_prime_gpu() {
        let cce = CategoricalCrossEntropy;
        let y_true = CpuTensor::new(vec![2, 3], vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).unwrap();

        let y_pred = CpuTensor::new(vec![2, 3], vec![0.7, 0.2, 0.1, 0.1, 0.3, 0.6]).unwrap();

        let res = cce.loss(&y_true, &y_pred).expect("CCE loss failed");
        let val = res.get_data()[0];

        let expected_loss = -(0.7_f32.ln() + 0.6_f32.ln());

        assert!(
            (val - expected_loss).abs() < 1e-6,
            "CCE loss mismatch: {} vs {}",
            val,
            expected_loss
        );

        let grad = cce
            .loss_prime(&y_true, &y_pred)
            .expect("CCE loss_prime failed");
        let g = grad.get_data();

        let expected_grads = vec![-0.15, 0.1, 0.05, 0.05, 0.15, -0.2];

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
        let cce = CategoricalCrossEntropy;
        let y_true = CpuTensor::new(vec![1, 2], vec![1.0, 0.0]).unwrap();
        let y_pred = CpuTensor::new(vec![1, 2], vec![0.0, 1.0]).unwrap(); // Total wrong prediction

        let res = cce
            .loss(&y_true, &y_pred)
            .expect("CCE stability test failed");
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
