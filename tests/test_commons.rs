#[cfg(test)]
mod tests {
    use iron_learn::commons::{add_bias_term, denormalize_features, normalize_features_mean_std};
    use iron_learn::CpuTensor;
    use iron_learn::Tensor;

    // Helper to create tensors easily
    fn get_cpu_tensor(shape: Vec<u32>, data: Vec<f32>) -> CpuTensor<f32> {
        CpuTensor::new(shape, data).unwrap()
    }

    #[test]
    fn test_add_bias_term() {
        // Input: 2x1 matrix [[5.0], [10.0]]
        let x = get_cpu_tensor(vec![2, 1], vec![5.0, 10.0]);
        let biased = add_bias_term(&x).expect("Failed to add bias");

        // Expected: 2x2 matrix [[1.0, 5.0], [1.0, 10.0]]
        assert_eq!(biased.get_shape(), &[2, 2]);
        let data = biased.get_data();
        assert_eq!(data, &[1.0, 5.0, 1.0, 10.0]);
    }

    #[test]
    fn test_normalization_reversibility() {
        // Create a 3x1 tensor with arbitrary data
        let original_data = vec![10.0, 20.0, 30.0];
        let x = get_cpu_tensor(vec![3, 1], original_data.clone());

        // 1. Normalize
        let (normalized, mean, std) = normalize_features_mean_std(&x);

        // 2. Denormalize
        let denormalized = denormalize_features(&normalized, &mean, &std);

        // Result should match original data (allowing for small floating point jitter)
        let result_data = denormalized.get_data();
        for (a, b) in original_data.iter().zip(result_data.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_normalize_mean_std_values() {
        // Data: [1, 3] -> Mean: 2, Var: ((1-2)^2 + (3-2)^2)/2 = 1, Std: 1
        let x = get_cpu_tensor(vec![2, 1], vec![1.0, 3.0]);
        let (normalized, mean, std) = normalize_features_mean_std(&x);

        assert_eq!(mean[0], 2.0);
        assert_eq!(std[0], 1.0);

        // Normalized values: (1-2)/1 = -1, (3-2)/1 = 1
        assert_eq!(normalized.get_data(), &[-1.0, 1.0]);
    }

    #[test]
    fn test_normalization_zero_std_division() {
        // If all values are the same (e.g., [5.0, 5.0]), std_dev is 0.
        // Your function handles this by returning 0.0 instead of NaN.
        let x = get_cpu_tensor(vec![2, 1], vec![5.0, 5.0]);
        let (normalized, _mean, _std) = normalize_features_mean_std(&x);

        // Should be [0.0, 0.0], not [NaN, NaN]
        for val in normalized.get_data() {
            assert_eq!(val, 0.0);
        }
    }
}
