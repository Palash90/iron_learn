mod tensor_ops {
    use iron_learn::Tensor;

    #[test]
    #[should_panic(expected = "TemporaryShapeRestriction")]
    fn test_new_panic_on_temp_restriction() {
        Tensor::new(Vec::new(), vec![1, 2, 3]).unwrap();
    }

    #[test]
    #[should_panic(expected = "DataError")]
    fn test_new_panic_on_data() {
        Tensor::new(vec![1, 2], vec![1, 2, 3]).unwrap();
    }

    #[test]
    pub fn add_i8() {
        let m1 = Tensor::<i8>::new(vec![1, 2], vec![1i8, 2i8]).unwrap();
        let m2 = Tensor::new(vec![1, 2], vec![3i8, 4i8]).unwrap();
        let result = Tensor::new(vec![1, 2], vec![4i8, 6i8]).unwrap();

        assert_eq!(result, (m1 + m2).unwrap());
    }

    #[test]
    pub fn add_i16() {
        let m1 = Tensor::new(vec![1, 2], vec![1i16, 2i16]).unwrap();
        let m2 = Tensor::new(vec![1, 2], vec![3i16, 4i16]).unwrap();
        let result = Tensor::new(vec![1, 2], vec![4i16, 6i16]).unwrap();

        assert_eq!(result, (m1 + m2).unwrap());
    }

    #[test]
    #[ignore]
    pub fn add_i32() {
        let m1 = Tensor::new(vec![1, 2, 2], vec![1, 2, 3, 4]).unwrap();
        let m2 = Tensor::new(vec![1, 2, 2], vec![5, 6, 7, 8]).unwrap();
        let result = Tensor::new(vec![1, 2, 2], vec![6, 8, 10, 12]).unwrap();

        assert_eq!(result, (m1 + m2).unwrap());
    }
}
