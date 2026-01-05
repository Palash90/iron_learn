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
}
