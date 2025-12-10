use iron_learn::CpuTensor;

#[test]
#[should_panic(expected = "TemporaryShapeRestriction")]
fn test_new_panic_on_shape_error() {
    CpuTensor::new(vec![1, 3, 5], vec![1, 2, 3]).unwrap();
}

#[test]
#[should_panic(expected = "ShapeError")]
fn test_new_panic_on_temp_restriction() {
    CpuTensor::new(Vec::new(), vec![1, 2, 3]).unwrap();
}

#[test]
#[should_panic(expected = "DataError")]
fn test_new_panic_on_data() {
    CpuTensor::new(vec![1, 2], vec![1, 2, 3]).unwrap();
}

#[test]
pub fn add_i8() {
    let m1 = CpuTensor::<i8>::new(vec![1, 2], vec![1i8, 2i8]).unwrap();
    let m2 = CpuTensor::new(vec![1, 2], vec![3i8, 4i8]).unwrap();
    let result = CpuTensor::new(vec![1, 2], vec![4i8, 6i8]).unwrap();

    assert_eq!(result, m1.add(&m2).unwrap());
}

#[test]
pub fn add_i16() {
    let m1 = CpuTensor::new(vec![1, 2], vec![1, 2]).unwrap();
    let m2 = CpuTensor::new(vec![1, 2], vec![3, 4]).unwrap();
    let result = CpuTensor::new(vec![1, 2], vec![4i16, 6i16]).unwrap();

    assert_eq!(result, m1.add(&m2).unwrap());
}

#[test]
#[ignore]
pub fn add_i32() {
    let m1 = CpuTensor::new(vec![1, 2, 2], vec![1, 2, 3, 4]).unwrap();
    let m2 = CpuTensor::new(vec![1, 2, 2], vec![5, 6, 7, 8]).unwrap();
    let result = CpuTensor::new(vec![1, 2, 2], vec![6, 8, 10, 12]).unwrap();

    assert_eq!(result, m1.add(&m2).unwrap());
}

#[test]
pub fn mul() {
    let m1 = CpuTensor::new(vec![1, 2], vec![1, 2]).unwrap();
    let m2 = CpuTensor::new(vec![2, 1], vec![3, 4]).unwrap();
    let result = CpuTensor::new(vec![1, 1], vec![11]).unwrap();

    assert_eq!(result, m1.mul(&m2).unwrap());
}

#[test]
pub fn sub() {
    let m1 = CpuTensor::new(vec![1, 2], vec![1, 2]).unwrap();
    let m2 = CpuTensor::new(vec![1, 2], vec![3, 4]).unwrap();
    let result = CpuTensor::new(vec![1, 2], vec![-2, -2]).unwrap();

    assert_eq!(result, m1.sub(&m2).unwrap());
}
