#[cfg(test)]
mod tests {
    // Assuming CpuTensor is your concrete implementation for testing
    use iron_learn::nn::*;
    use iron_learn::CpuTensor;
    use iron_learn::Tensor;

    #[test]
    fn test_get_activations_sigmoid() {
        let (act, prime) = get_activations::<CpuTensor<f32>, f32>(&LayerType::Sigmoid);

        // Input tensor [0.0] -> Sigmoid(0) = 0.5
        let input = CpuTensor::new(vec![1], vec![0.0]).unwrap();
        let result = act(&input).unwrap();

        assert!((result.get_data()[0] - 0.5).abs() < 1e-6);

        // Derivative test: sigmoid_prime(output) = output * (1 - output)
        // If output is 0.5, prime is 0.5 * (1 - 0.5) = 0.25
        let prime_result = prime(&result).unwrap();
        assert!((prime_result.get_data()[0] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_get_activations_linear() {
        let (act, prime) = get_activations::<CpuTensor<f32>, f32>(&LayerType::Linear);

        let input = CpuTensor::new(vec![2], vec![5.0, -2.0]).unwrap();

        // Linear activation should return the same values
        let result = act(&input).unwrap();
        assert_eq!(result.get_data(), vec![5.0, -2.0]);

        // Linear derivative should always return 1s regardless of input
        let prime_result = prime(&result).unwrap();
        assert_eq!(prime_result.get_data(), vec![1.0, 1.0]);
    }

    #[test]
    fn test_get_activations_tanh() {
        let (act, prime) = get_activations::<CpuTensor<f32>, f32>(&LayerType::Tanh);

        let input = CpuTensor::new(vec![1], vec![0.0]).unwrap();
        let result = act(&input).unwrap();

        // Tanh(0) = 0
        assert!((result.get_data()[0]).abs() < 1e-6);

        // Tanh prime: 1 - tanh(x)^2. Since tanh(0)=0, prime is 1 - 0 = 1
        let prime_result = prime(&result).unwrap();
        assert!((prime_result.get_data()[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_single_row_logic() {
        // Input is 1 row, 3 classes
        let input = CpuTensor::<f32>::new(vec![1, 3], vec![0.0, 1.0, 2.0]).unwrap();
        let result = softmax(&input).expect("Softmax failed on single row");
        
        let data = result.get_data();
        let sum: f32 = data.iter().sum();

        // 1. Sum must be 1.0 (since sum() only sees this one row)
        assert!((sum - 1.0).abs() < 1e-6);

        // 2. Relative probabilities: e^2 is much larger than e^1 or e^0
        // Expected: [0.09, 0.24, 0.67] roughly
        assert!(data[2] > data[1]);
        assert!(data[1] > data[0]);
    }

    #[test]
    fn test_softmax_prime_pass_through() {
        // Verifies the identity derivative works for a single row
        let output =  CpuTensor::<f32>::new(vec![1, 3], vec![0.2, 0.7, 0.1]).unwrap();
        let grad = softmax_prime(&output).expect("Softmax prime failed");
        
        let shape = grad.get_shape();
        let data = grad.get_data();

        assert_eq!(shape, &vec![1, 3]);
        assert!(data.iter().all(|&x| x == 1.0), "Prime must return 1.0 for pass-through");
    }

    #[test]
    fn test_softmax_large_values_fail_check() {
        // This test serves as a reminder for your "Numerical Stability" future task.
        // If this panics/returns Inf, you know you still need max-subtraction.
        let input = CpuTensor::<f32>::new(vec![1, 2], vec![100.0, 100.0]).unwrap();
        let result = softmax(&input);
        
        if let Ok(res) = result {
            let data = res.get_data();
            assert!(!data[0].is_nan(), "Numerical overflow detected! e^100 is too large.");
        }
    }
}
