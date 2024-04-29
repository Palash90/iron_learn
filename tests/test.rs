use iron_learn::Tensor;

#[test]
pub fn add_i8() {
    let m1 = Tensor::<i8>::new(vec![1u32, 2u32], vec![1i8, 2i8]).unwrap();
    let m2 = Tensor::new(vec![1u32, 2u32], vec![3i8, 4i8]).unwrap();
    let result = Tensor::new(vec![1u32, 2u32], vec![4i8, 6i8]);

    println!("Hello Test");

    assert_eq!(result, m1 + m2);
}


#[test]
pub fn add_i16() {
    let m1 = Tensor::new(vec![1u32, 2u32], vec![1i16, 2i16]).unwrap();
    let m2 = Tensor::new(vec![1u32, 2u32], vec![3i16, 4i16]).unwrap();
    let result = Tensor::new(vec![1u32, 2u32], vec![4i16, 6i16]);

    assert_eq!(result, m1 + m2);
}
