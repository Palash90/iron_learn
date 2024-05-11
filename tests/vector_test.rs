use iron_learn::Vector;

#[test]
#[should_panic(expected = "VectorShapeError")]
fn test_new_panic_on_shape() {
    Vector::new(vec![1, 2, 3], vec![1, 2, 3]).unwrap();
}

#[test]
pub fn add_i8() {
    let m1 = Vector::<i8>::new(vec![1], vec![1i8]).unwrap();
    let m2 = Vector::new(vec![1], vec![3i8]).unwrap();
    let result = Vector::new(vec![1], vec![4i8]).unwrap();

    assert_eq!(result, (m1 + m2).unwrap());
}

#[test]
pub fn add_i16() {
    let m1 = Vector::new(vec![1], vec![1i16]).unwrap();
    let m2 = Vector::new(vec![1], vec![1i16]).unwrap();
    let result = Vector::new(vec![1], vec![2i16]).unwrap();

    assert_eq!(result, (m1 + m2).unwrap());
}

#[test]
pub fn add_i32() {
    let m1 = Vector::new(vec![1], vec![1]).unwrap();
    let m2 = Vector::new(vec![1], vec![3]).unwrap();
    let result = Vector::new(vec![1], vec![4]).unwrap();

    assert_eq!(result, (m1 + m2).unwrap());
}

#[test]
pub fn mul_i32() {
    let m1 = Vector::new(vec![1], vec![1]).unwrap();
    let m2 = Vector::new(vec![1], vec![3]).unwrap();

    assert_eq!(3, (m1 * m2).unwrap());
}

#[test]
pub fn mul_2_cols() {
    let m1 = Vector::new(vec![2], vec![1, 2]).unwrap();
    let m2 = Vector::new(vec![2], vec![3, 4]).unwrap();

    assert_eq!(11, (m1 * m2).unwrap());
}
