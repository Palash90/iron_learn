#[cfg(test)]
mod exhaustive_tests {
    use iron_learn::Complex;
    use iron_learn::Numeric;
    use iron_learn::SignedNumeric;

    /// 1. Generic Axiom Testing
    /// Verifies that any type T implementing Numeric follows basic mathematical identities.
    fn verify_numeric_identities<T: Numeric>(val: T) {
        // Identity: x + 0 = x
        assert_eq!(val + T::zero(), val, "Addition identity failed for {}", val);
        // Identity: x * 1 = x
        assert_eq!(
            val * T::one(),
            val,
            "Multiplication identity failed for {}",
            val
        );
        // Identity: x - x = 0
        assert_eq!(
            val - val,
            T::zero(),
            "Subtraction self-nullification failed for {}",
            val
        );
        // Identity: x / 1 = x
        assert_eq!(val / T::one(), val, "Division identity failed for {}", val);
    }

    #[test]
    fn test_all_identities() {
        verify_numeric_identities(100i8);
        verify_numeric_identities(5000u32);
        verify_numeric_identities(std::f64::consts::PI);
        verify_numeric_identities(Complex::new(5.0, -2.0));
    }

    /// 2. Integer Boundary Tests
    #[test]
    fn test_integer_boundaries() {
        // Test i8 (Range: -128 to 127)
        assert_eq!(i8::from_f64(127.0), 127);
        assert_eq!(i8::from_f64(-128.0), -128);

        // Test u8 (Range: 0 to 255)
        assert_eq!(u8::from_u32(255), 255);
        // Note: from_u32 uses .unwrap(), so u8::from_u32(300) would panic.
    }

    /// 3. Casting Precision & Truncation
    #[test]
    fn test_casting_behavior() {
        // Floating point to Integer (Truncation check)
        assert_eq!(i32::from_f64(10.99), 10);
        assert_eq!(i32::from_f64(-10.99), -10);

        // Integer to Float
        let big_val: i128 = 1_000_000_000_000;
        assert_eq!(big_val.f64(), 1_000_000_000_000.0);
    }

    /// 4. FloatingPoint Specifics (NaN, Infinity, Precision)
    #[test]
    fn test_fp_special_cases() {
        let nan = f64::NAN;
        let inf = f64::INFINITY;

        assert!(nan.f64().is_nan());
        assert!(inf.f64().is_infinite());

        // Transcendental checks
        let angle = f64::from_f64(std::f64::consts::PI);
        assert!((angle.sin() - 0.0).abs() < 1e-10);
        assert!((angle.cos() + 1.0).abs() < 1e-10);
    }

    /// 5. Complex Number Arithmetic
    #[test]
    fn test_complex_deep_math() {
        let a = Complex::new(2.0, 3.0);
        let b = Complex::new(1.0, -1.0);

        // (2+3i) + (1-i) = 3+2i
        assert_eq!(a + b, Complex::new(3.0, 2.0));
        // (2+3i) * (1-i) = 2 - 2i + 3i - 3i^2 = 2 + i + 3 = 5 + i
        assert_eq!(a * b, Complex::new(5.0, 1.0));
    }

    /// 6. SignedNumeric Negation
    #[test]
    fn test_negation_all_types() {
        fn test_neg<T: SignedNumeric>(val: T, expected: T) {
            assert_eq!(-val, expected);
        }

        test_neg(10i32, -10i32);
        test_neg(5.5f64, -5.5f64);
        test_neg(Complex::new(1.0, 2.0), Complex::new(-1.0, -2.0));
    }
}
