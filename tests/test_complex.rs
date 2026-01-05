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

    // Test Negation
    let r = -a;
    assert_eq!(Complex::new(-1.0, -2.0), r);
}

#[test]
fn test_complex_display_positive_imaginary() {
    let z = Complex::new(3.0, 4.0);
    // This triggers the fmt function logic for self.imaginary >= 0.0
    assert_eq!(format!("{}", z), "3 + 4i");
}

#[test]
fn test_complex_display_negative_imaginary() {
    let z = Complex::new(2.0, -1.0);
    // This triggers the fmt function logic for the else block
    // Note: -(-1.0) becomes 1.0 in your write! macro
    assert_eq!(format!("{}", z), "2 - 1i");
}

#[test]
fn test_complex_display_zero_imaginary() {
    let z = Complex::new(5.0, 0.0);
    // 0.0 is >= 0.0, so it should use the '+' branch
    assert_eq!(format!("{}", z), "5 + 0i");
}

#[test]
fn test_complex_display_negative_real() {
    let z = Complex::new(-1.5, -2.5);
    assert_eq!(format!("{}", z), "-1.5 - 2.5i");
}
