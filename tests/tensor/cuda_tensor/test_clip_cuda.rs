#[cfg(test)]
mod tests {
    use iron_learn::GpuTensor;
    use iron_learn::Tensor;

    use iron_learn::init_gpu;

    #[test]
    fn test_cuda_clip_basic_bounds() {
        let _ = init_gpu();

        let input = vec![-10.0, 0.0, 5.0, 10.0, 20.0];
        let min = 0.0;
        let max = 10.0;

        let expected = vec![0.0, 0.0, 5.0, 10.0, 10.0];

        let tensor = GpuTensor::<f32>::new(vec![5], input).unwrap();
        let clipped = tensor.clip(min, max).unwrap();

        assert_eq!(clipped.get_data(), expected);
    }

    #[test]
    fn test_cuda_clip_all_below() {
        let _ = init_gpu();

        let input = vec![-5.0, -2.0, -1.0];
        let min = 0.0;
        let max = 10.0;
        let expected = vec![0.0, 0.0, 0.0];

        let tensor = GpuTensor::<f32>::new(vec![3], input).unwrap();
        let clipped = tensor.clip(min, max).unwrap();
        assert_eq!(clipped.get_data(), expected);
    }

    #[test]
    fn test_cuda_clip_all_above() {
        let _ = init_gpu();

        let input = vec![15.0, 20.0, 100.0];
        let min = 0.0;
        let max = 10.0;
        let expected = vec![10.0, 10.0, 10.0];

        let tensor = GpuTensor::<f32>::new(vec![3], input).unwrap();
        let clipped = tensor.clip(min, max).unwrap();
        assert_eq!(clipped.get_data(), expected);
    }

    #[test]
    fn test_cuda_clip_identical_bounds() {
        let _ = init_gpu();

        let input = vec![1.0, 2.0, 3.0];
        let min = 5.0;
        let max = 5.0;
        let expected = vec![5.0, 5.0, 5.0];

        let tensor = GpuTensor::<f32>::new(vec![3], input).unwrap();
        let clipped = tensor.clip(min, max).unwrap();
        assert_eq!(clipped.get_data(), expected);
    }

    #[test]
    fn test_cuda_clip_with_nans() {
        let _ = init_gpu();

        let input = vec![f32::NAN, 5.0];
        let min = 0.0;
        let max = 10.0;

        let tensor = GpuTensor::<f32>::new(vec![2], input).unwrap();
        let clipped = tensor.clip(min, max).unwrap();

        assert!(clipped.get_data()[0].is_nan());
        assert_eq!(clipped.get_data()[1], 5.0);
    }
}
