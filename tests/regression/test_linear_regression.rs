#[cfg(test)]
mod tests {
    use iron_learn::CpuTensor;
    use iron_learn::Tensor;
    use iron_learn::linear_regression::linear_regression;
    use iron_learn::linear_regression::predict_linear;


    fn get_cpu_tensor(shape: Vec<u32>, data: Vec<f32>) -> CpuTensor<f32> {
        CpuTensor::new(shape, data).unwrap()
    }

    #[test]
    fn test_predict_linear_multi_dim() {
        // X is 2x2, W is 2x1 -> Output should be 2x1
        // [[1, 2], [3, 4]] * [[0.5], [1.0]] = [[2.5], [5.5]]
        let x = get_cpu_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let w = get_cpu_tensor(vec![2, 1], vec![0.5, 1.0]);

        let prediction = predict_linear(&x, &w).expect("Prediction failed");

        assert_eq!(prediction.get_shape(), &vec![2, 1]);
        assert_eq!(prediction.get_data().clone(), vec![2.5, 5.5]);
    }

    #[test]
    fn test_linear_regression_full_convergence() {
        // Goal: Learn y = 1 + 2x ---> Bias comes first in Iron Learn
        // Inputs (X) are [1, 2, 3]
        // Outputs (Y) are [3, 5, 7]
        let x = get_cpu_tensor(vec![3, 1], vec![1.0, 2.0, 3.0]);
        let y = get_cpu_tensor(vec![3, 1], vec![3.0, 5.0, 7.0]);

        // Initial weights for [weight, bias] because linear_regression adds bias term
        let initial_w = get_cpu_tensor(vec![2, 1], vec![0.0, 0.0]);

        let lr = 0.05;
        let epochs = 500;

        let trained_w = linear_regression(&x, &y, initial_w, lr, epochs).expect("Training failed");

        let final_weights = trained_w.get_data();

        // final_weights[0] should be bias (~1.0)
        // final_weights[1] should be weight (~2.0)
        assert!(
            (final_weights[0] - 1.0).abs() < 0.1,
            "Weight did not converge"
        );
        assert!(
            (final_weights[1] - 2.0).abs() < 0.1,
            "Bias did not converge"
        );
    }

    #[test]
    fn test_linear_regression_mismatched_y() {
        // X has 2 samples, Y has 3 samples. This should fail.
        let x = get_cpu_tensor(vec![2, 1], vec![1.0, 2.0]);
        let y = get_cpu_tensor(vec![3, 1], vec![2.0, 4.0, 6.0]);
        let w = get_cpu_tensor(vec![2, 1], vec![0.0, 0.0]);

        let result = linear_regression(&x, &y, w, 0.1, 10);
        assert!(
            result.is_err(),
            "Should error when X and Y sample counts differ"
        );
    }
}
