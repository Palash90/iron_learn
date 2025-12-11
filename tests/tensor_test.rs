use iron_learn::CpuTensor;
use iron_learn::Tensor;
use iron_learn::tensor::math::TensorMath;

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

    assert_eq!(result, (m1 + m2).unwrap());
}

#[test]
pub fn add_i16() {
    let m1 = CpuTensor::new(vec![1, 2], vec![1i16, 2i16]).unwrap();
    let m2 = CpuTensor::new(vec![1, 2], vec![3i16, 4i16]).unwrap();
    let result = CpuTensor::new(vec![1, 2], vec![4i16, 6i16]).unwrap();

    assert_eq!(result, (m1 + m2).unwrap());
}

#[test]
#[ignore]
pub fn add_i32() {
    let m1 = CpuTensor::new(vec![1, 2, 2], vec![1, 2, 3, 4]).unwrap();
    let m2 = CpuTensor::new(vec![1, 2, 2], vec![5, 6, 7, 8]).unwrap();
    let result = CpuTensor::new(vec![1, 2, 2], vec![6, 8, 10, 12]).unwrap();

    assert_eq!(result, (m1 + m2).unwrap());
}

#[test]
pub fn sub() {
    let m1: CpuTensor<i32> = CpuTensor::new(vec![1, 2], vec![1, 2]).unwrap();
    let m2 = CpuTensor::new(vec![1, 2], vec![3, 4]).unwrap();
    let result = CpuTensor::new(vec![1, 2], vec![-2, -2]).unwrap();

    assert_eq!(result, (m1 - m2).unwrap());
}

#[test]
fn test_shape_mismatch_error() {
    // 2x4 Matrix
    let matrix = CpuTensor::new(vec![2, 4], vec![1.0; 8]).unwrap();
    // 1D Vector (length 3)
    let vector = CpuTensor::new(vec![3], vec![1.0; 3]).unwrap();

    let result = matrix + vector;

    // Check that the result is an Err variant
    assert!(result.is_err());

    // Optional: Check the exact error message
    // assert_eq!(result.unwrap_err(), "ShapeMismatch:The dimensions of two matrices are not compatible for addition/subtraction- [2, 4] [3]".to_string());
}

#[test]
fn test_2d_incompatible_shapes_error() {
    // 3x4 Matrix
    let m1 = CpuTensor::new(vec![3, 4], vec![1.0; 12]).unwrap();
    // 2x5 Matrix
    let m2 = CpuTensor::new(vec![2, 5], vec![1.0; 10]).unwrap();

    let result = m1 + m2;

    // Check that the result is an Err variant
    assert!(result.is_err());
}

#[test]
pub fn transpose() {
    let m = CpuTensor::new(vec![2, 2], vec![1, 2, 3, 4]).unwrap();
    let result = CpuTensor::new(vec![2, 2], vec![1, 3, 2, 4]).unwrap();

    assert_eq!(result, m.t().unwrap());

    let m = CpuTensor::new(vec![2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();
    let result = CpuTensor::new(vec![3, 2], vec![1, 4, 2, 5, 3, 6]).unwrap();

    assert_eq!(result, m.t().unwrap());

    let m = CpuTensor::new(vec![6], vec![1, 2, 3, 4, 5, 6]).unwrap();
    let result = CpuTensor::new(vec![6], vec![1, 2, 3, 4, 5, 6]).unwrap();

    assert_eq!(result, m.t().unwrap());

    let m = CpuTensor::new(vec![3, 3], vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();
    let result = CpuTensor::new(vec![3, 3], vec![1, 4, 7, 2, 5, 8, 3, 6, 9]).unwrap();

    assert_eq!(result, m.t().unwrap());

    let m = CpuTensor::new(vec![3, 1], vec![1, 2, 3]).unwrap();
    let result = CpuTensor::new(vec![1, 3], vec![1, 2, 3]).unwrap();

    assert_eq!(result, m.t().unwrap());
}

fn sigmoid(x: f64) -> f64 {
    f64::exp(x) / (1.0 + f64::exp(x))
}

#[test]
pub fn fn_test() {
    let m1 = CpuTensor::new(vec![1, 2], vec![1, 2]).unwrap();

    let result = CpuTensor::new(vec![1, 2], vec![f64::sin(1.0), f64::sin(2.0)]).unwrap();
    assert_eq!(result, CpuTensor::sin(&m1).unwrap());

    let result = CpuTensor::new(vec![1, 2], vec![f64::exp(1.0), f64::exp(2.0)]).unwrap();
    assert_eq!(result, CpuTensor::exp(&m1).unwrap());

    let result = CpuTensor::new(vec![1, 2], vec![f64::cos(1.0), f64::cos(2.0)]).unwrap();
    assert_eq!(result, CpuTensor::cos(&m1).unwrap());

    let result = CpuTensor::new(vec![1, 2], vec![f64::tan(1.0), f64::tan(2.0)]).unwrap();
    assert_eq!(result, CpuTensor::tan(&m1).unwrap());

    let result = CpuTensor::new(vec![1, 2], vec![f64::tanh(1.0), f64::tanh(2.0)]).unwrap();
    assert_eq!(result, CpuTensor::tanh(&m1).unwrap());

    let result = CpuTensor::new(vec![1, 2], vec![f64::log10(1.0), f64::log10(2.0)]).unwrap();
    assert_eq!(result, CpuTensor::log(&m1).unwrap());

    let result = CpuTensor::new(vec![1, 2], vec![f64::ln(1.0), f64::ln(2.0)]).unwrap();
    assert_eq!(result, CpuTensor::ln(&m1).unwrap());

    let result = CpuTensor::new(vec![1, 2], vec![sigmoid(1.0), sigmoid(2.0)]).unwrap();
    assert_eq!(result, CpuTensor::sigmoid(&m1).unwrap());
}
