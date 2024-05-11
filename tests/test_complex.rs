use iron_learn::Complex;

#[test]
fn test_complex() {
    let a = Complex::new(1.0, 2.0);
    let b = Complex::new(3.0, 4.0);

    // Test Addition
    let r = a + b;
    assert_eq!(Complex::new(4.0, 6.0), r);

    let r = b + a;
    assert_eq!(Complex::new(4.0, 6.0), r);

    // Test Subtraction
    let r = a - b;
    assert_eq!(Complex::new(-2.0, -2.0), r);

    let r = b - a;
    assert_eq!(Complex::new(2.0, 2.0), r);

    // Test Multiplication
    let r = a * b;
    assert_eq!(Complex::new(-5.0, 10.0), r);

    let r = b * a;
    assert_eq!(Complex::new(-5.0, 10.0), r);

    // Test Division
    let r = a / b;
    assert_eq!(Complex::new(11.0 / 25.0, 2.0 / 25.0), r);

    let r = b / a;
    assert_eq!(Complex::new(11.0 / 5.0, -(2.0 / 5.0)), r);
}
