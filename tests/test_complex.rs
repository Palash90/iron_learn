use iron_learn::complex::Complex;

#[test]
fn test_complex() {
    let a = Complex {
        real: 1.0,
        imaginary: 2.0,
    };
    let b = Complex {
        real: 3.0,
        imaginary: 4.0,
    };

    // Test Addition
    let r = a + b;
    assert_eq!(
        Complex {
            real: 4.0,
            imaginary: 6.0
        },
        r
    );

    let r = b + a;
    assert_eq!(
        Complex {
            real: 4.0,
            imaginary: 6.0
        },
        r
    );

    // Test Subtraction
    let r = a - b;
    assert_eq!(
        Complex {
            real: -2.0,
            imaginary: -2.0
        },
        r
    );

    let r = b - a;
    assert_eq!(
        Complex {
            real: 2.0,
            imaginary: 2.0
        },
        r
    );

    // Test Multiplication
    let r = a * b;
    assert_eq!(
        Complex {
            real: -5.0,
            imaginary: 10.0
        },
        r
    );

    let r = b * a;
    assert_eq!(
        Complex {
            real: -5.0,
            imaginary: 10.0
        },
        r
    );

    // Test Division
    let r = a / b;
    assert_eq!(
        Complex {
            real: 11.0 / 25.0,
            imaginary: 2.0 / 25.0
        },
        r
    );

    let r = b / a;
    assert_eq!(
        Complex {
            real: 11.0 / 5.0,
            imaginary: -(2.0 / 5.0)
        },
        r
    );
}
