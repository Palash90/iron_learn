mod matrix_add {
    use iron_learn::Matrix;

    #[test]
    pub fn add_i8() {
        let m1 = Matrix::<i8>::new(vec![1u32, 2u32], vec![1i8, 2i8]).unwrap();
        let m2 = Matrix::new(vec![1u32, 2u32], vec![3i8, 4i8]).unwrap();
        let result = Matrix::new(vec![1u32, 2u32], vec![4i8, 6i8]).unwrap();

        println!("Hello Test");

        assert_eq!(result, (m1 + m2).unwrap());
    }

    #[test]
    pub fn add_i16() {
        let m1 = Matrix::new(vec![1u32, 2u32], vec![1i16, 2i16]).unwrap();
        let m2 = Matrix::new(vec![1u32, 2u32], vec![3i16, 4i16]).unwrap();
        let result = Matrix::new(vec![1u32, 2u32], vec![4i16, 6i16]).unwrap();

        assert_eq!(result, (m1 + m2).unwrap());
    }

    #[test]
    pub fn add_i32() {
        let m1 = Matrix::new(vec![1u32, 2u32], vec![1, 2]).unwrap();
        let m2 = Matrix::new(vec![1u32, 2u32], vec![3, 4]).unwrap();
        let result = Matrix::new(vec![1u32, 2u32], vec![4, 6]).unwrap();

        assert_eq!(result, (m1 + m2).unwrap());
    }
}
