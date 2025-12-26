#[cfg(test)]
mod tests {
    use iron_learn::CpuTensor;
    use iron_learn::Tensor;

    #[test]
    fn test_clip_basic_bounds() {
        // Setup data with values below, inside, and above range
        let input = vec![-10.0, 0.0, 5.0, 10.0, 20.0];
        let min = 0.0;
        let max = 10.0;

        // Expected: [-10 -> 0], [0 -> 0], [5 -> 5], [10 -> 10], [20 -> 10]
        let expected = vec![0.0, 0.0, 5.0, 10.0, 10.0];

        let tensor = CpuTensor::new(vec![5], input).unwrap();
        let clipped = tensor.clip(min, max).unwrap();

        assert_eq!(clipped.get_data(), expected);
    }

    #[test]
    fn test_clip_all_below() {
        let input = vec![-5.0, -2.0, -1.0];
        let min = 0.0;
        let max = 10.0;
        let expected = vec![0.0, 0.0, 0.0];

        let tensor = CpuTensor::new(vec![3], input).unwrap();
        let clipped = tensor.clip(min, max).unwrap();
        assert_eq!(clipped.get_data(), expected);
    }

    #[test]
    fn test_clip_all_above() {
        let input = vec![15.0, 20.0, 100.0];
        let min = 0.0;
        let max = 10.0;
        let expected = vec![10.0, 10.0, 10.0];

        let tensor = CpuTensor::new(vec![3], input).unwrap();
        let clipped = tensor.clip(min, max).unwrap();
        assert_eq!(clipped.get_data(), expected);
    }

    #[test]
    fn test_clip_empty_data() {
        let input: Vec<f32> = vec![];
        let tensor = CpuTensor::new(vec![0], input).unwrap();
        let clipped = tensor.clip(0.0, 10.0).unwrap();

        assert!(clipped.get_data().is_empty());
    }

    #[test]
    fn test_clip_identical_bounds() {
        // If min == max, everything should become that value
        let input = vec![1.0, 2.0, 3.0];
        let min = 5.0;
        let max = 5.0;
        let expected = vec![5.0, 5.0, 5.0];

        let tensor = CpuTensor::new(vec![3], input).unwrap();
        let clipped = tensor.clip(min, max).unwrap();
        assert_eq!(clipped.get_data(), expected);
    }

    #[test]
    fn test_clip_with_nans() {
        // Note: Floating point NaNs usually fail PartialOrd comparisons
        // Depending on your requirements, you might want to handle this explicitly
        let input = vec![f32::NAN, 5.0];
        let min = 0.0;
        let max = 10.0;

        let tensor = CpuTensor::new(vec![2], input).unwrap();
        let clipped = tensor.clip(min, max).unwrap();

        // In the standard if/else logic, NaN < min is false and NaN > max is false
        // So NaN usually stays NaN.
        assert!(clipped.get_data()[0].is_nan());
        assert_eq!(clipped.get_data()[1], 5.0);
    }
}
