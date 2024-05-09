mod vector_ops {
    use iron_learn::Vector;

    #[test]
    #[should_panic(expected = "VectorShapeError")]
    fn test_new_panic_on_shape() {
        Vector::new(vec![1u32, 2u32, 3u32], vec![1, 2, 3]).unwrap();
    }

    #[test]
    pub fn add_i8() {
        let m1 = Vector::<i8>::new(vec![1u32], vec![1i8]).unwrap();
        let m2 = Vector::new(vec![1u32], vec![3i8]).unwrap();
        let result = Vector::new(vec![1u32], vec![4i8]).unwrap();

        assert_eq!(result, (m1 + m2).unwrap());
    }

    #[test]
    pub fn add_i16() {
        let m1 = Vector::new(vec![1u32], vec![1i16]).unwrap();
        let m2 = Vector::new(vec![1u32], vec![1i16]).unwrap();
        let result = Vector::new(vec![1u32], vec![2i16]).unwrap();

        assert_eq!(result, (m1 + m2).unwrap());
    }

    #[test]
    pub fn add_i32() {
        let m1 = Vector::new(vec![1u32], vec![1]).unwrap();
        let m2 = Vector::new(vec![1u32], vec![3]).unwrap();
        let result = Vector::new(vec![1u32], vec![4]).unwrap();

        assert_eq!(result, (m1 + m2).unwrap());
    }
}