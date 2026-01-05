#[cfg(test)]
mod logistic_tests {
    use iron_learn::logistic_regression::logistic_regression;
    use iron_learn::logistic_regression::predict_logistic;
    use iron_learn::CpuTensor;
    use iron_learn::Tensor;

    fn get_cpu_tensor(shape: Vec<u32>, data: Vec<f32>) -> CpuTensor<f32> {
        CpuTensor::new(shape, data).unwrap()
    }

    #[test]
    fn test_predict_logistic_thresholding() {
        let w = get_cpu_tensor(vec![2, 1], vec![2.0, -10.0]);

        let x = get_cpu_tensor(vec![2, 1], vec![6.0, 2.0]);

        let predictions = predict_logistic(&x, &w).expect("Prediction failed");

        assert_eq!(predictions.get_data().clone(), vec![0.0, 0.0]);
    }

    #[test]
    fn test_logistic_regression_convergence() {
        let x = get_cpu_tensor(vec![4, 1], vec![0.0, 1.0, 10.0, 11.0]);
        let y = get_cpu_tensor(vec![4, 1], vec![0.0, 0.0, 1.0, 1.0]);

        let initial_w = get_cpu_tensor(vec![2, 1], vec![0.0, 0.0]);
        let lr = 0.1;
        let epochs = 1000;

        let trained_w =
            logistic_regression(&x, &y, initial_w, lr, epochs).expect("Training failed");

        let final_preds = predict_logistic(&x, &trained_w).expect("Final prediction failed");

        assert_eq!(final_preds.get_data().clone(), vec![0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_logistic_prediction_shapes() {
        let x = get_cpu_tensor(vec![3, 1], vec![1.0, 2.0, 3.0]);
        let w = get_cpu_tensor(vec![2, 1], vec![0.1, 0.1]);

        let result = predict_logistic(&x, &w).unwrap();

        assert_eq!(result.get_shape(), &vec![3, 1]);
        assert_eq!(result.get_data().len(), 3);
    }
}
