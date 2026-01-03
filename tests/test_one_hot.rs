#[cfg(test)]
mod tests {
    use iron_learn::{
        one_hot::{one_hot_decode, one_hot_encode},
        CpuTensor, Tensor,
    };

    #[cfg(feature = "cuda")]
    use iron_learn::GpuTensor;
#[cfg(feature = "cuda")]
    use iron_learn::init_gpu;

    // Assuming these are your two implementations

    /// The Super Test Logic: Tests encoding, decoding, and edge cases for any Tensor implementation.
    fn run_tensor_suite<U>()
    where
        U: Tensor<f32>,
    {
        // 1. Test Standard Roundtrip
        let labels = vec![0, 1, 2, 1];
        let num_classes = 3;

        let encoded: U =
            one_hot_encode(&labels, num_classes).expect("Encoding failed in super test");

        let decoded = one_hot_decode(&encoded).expect("Decoding failed in super test");

        assert_eq!(
            labels,
            decoded,
            "Roundtrip failed for {}",
            std::any::type_name::<U>()
        );

        // 2. Test Shape Integrity
        assert_eq!(encoded.get_shape(), &vec![4, 3]);

        // 3. Test Out of Bounds Handling
        let bad_labels = vec![5];
        let bad_res: Result<U, String> = one_hot_encode(&bad_labels, 3);
        assert!(
            bad_res.is_err(),
            "Should have failed on out-of-bounds label"
        );

        // 4. Test Single Class Case
        let single_labels = vec![0, 0];
        let single_encoded: U = one_hot_encode(&single_labels, 1).unwrap();
        assert_eq!(single_encoded.get_data(), vec![1.0, 1.0]);
    }

    #[test]
    fn test_cpu_implementation() {
        run_tensor_suite::<CpuTensor<f32>>();
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_gpu_implementation() {
        let _ = init_gpu();
        run_tensor_suite::<GpuTensor<f32>>();
    }
}
