use iron_learn::Complex;
use iron_learn::CpuTensor;
use iron_learn::Tensor;
use iron_learn::tensor::math::TensorMath;

#[test]
fn test_complex_tensor_mul() {
    let a = Complex::new(1.0, 2.0);
    let b = Complex::new(3.0, 4.0);
    let c = Complex::new(5.0, 6.0);
    let d = Complex::new(7.0, 8.0);

    let m1 = CpuTensor::new(vec![2, 2], vec![a, b, c, d]).unwrap();
    let m2 = CpuTensor::new(vec![2, 2], vec![a, c, b, d]).unwrap();

    let result = (m1 * m2).unwrap();

    let r1 = Complex::new(-10.0, 28.0);
    let r2 = Complex::new(-18.0, 68.0);
    let r3 = Complex::new(-18.0, 68.0);
    let r4 = Complex::new(-26.0, 172.0);

    let expected = CpuTensor::new(vec![2, 2], vec![r1, r2, r3, r4]).unwrap();
    assert_eq!(expected, result);
}

#[test]
fn test_complex_tensor_add() {
    let a = Complex::new(1.0, 2.0);
    let b = Complex::new(3.0, 4.0);
    let c = Complex::new(5.0, 6.0);
    let d = Complex::new(7.0, 8.0);

    let m1 = CpuTensor::new(vec![2, 2], vec![a, b, c, d]).unwrap();
    let m2 = CpuTensor::new(vec![2, 2], vec![a, c, b, d]).unwrap();

    let result = (m1 + m2).unwrap();

    let r1 = Complex::new(2.0, 4.0);
    let r2 = Complex::new(8.0, 10.0);
    let r3 = Complex::new(8.0, 10.0);
    let r4 = Complex::new(14.0, 16.0);

    let expected = CpuTensor::new(vec![2, 2], vec![r1, r2, r3, r4]).unwrap();
    assert_eq!(expected, result);
}
