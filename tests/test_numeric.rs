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

#[cfg(test)]
mod tests {
    use iron_learn::Complex;
    use iron_learn::Numeric;

    /// Macro to generate tests for standard Numeric types
    macro_rules! test_numeric_impl {
        ($t:ty, $name:ident, $is_float:expr) => {
            #[test]
            fn $name() {
                let val: $t = <$t as Numeric>::one();
                let zero: $t = <$t as Numeric>::zero();

                // Test zero and one
                assert_eq!(val, 1 as $t);
                assert_eq!(zero, 0 as $t);

                // Test f32/f64 conversions
                assert_eq!(val.f32(), 1.0_f32);
                assert_eq!(val.f64(), 1.0_f64);

                // Test from_u32 and from_f64
                // We use 10 to avoid overflow panics in try_into blocks
                assert_eq!(<$t as Numeric>::from_u32(10), 10 as $t);
                assert_eq!(<$t as Numeric>::from_f64(10.0), 10 as $t);
            }
        };
    }

    // Call the macro for every integer and float type implemented in the file
    test_numeric_impl!(i8, test_i8, false);
    test_numeric_impl!(i16, test_i16, false);
    test_numeric_impl!(i32, test_i32, false);
    test_numeric_impl!(i64, test_i64, false);
    test_numeric_impl!(i128, test_i128, false);
    test_numeric_impl!(isize, test_isize, false);
    test_numeric_impl!(u8, test_u8, false);
    test_numeric_impl!(u16, test_u16, false);
    test_numeric_impl!(u32, test_u32, false);
    test_numeric_impl!(u64, test_u64, false);
    test_numeric_impl!(u128, test_u128, false);
    test_numeric_impl!(usize, test_usize, false);
    test_numeric_impl!(f32, test_f32, true);
    test_numeric_impl!(f64, test_f64, true);

    #[test]
    fn test_complex_numeric() {
        let _c = Complex::new(5.0, 0.0);
        assert_eq!(Complex::zero(), Complex::new(0.0, 0.0));
        assert_eq!(Complex::one(), Complex::new(1.0, 0.0));
        assert_eq!(Complex::from_u32(10), Complex::new(10.0, 0.0));
        assert_eq!(Complex::from_f64(10.5), Complex::new(10.5, 0.0));
    }

    #[test]
    #[should_panic(expected = "InvalidOperation")]
    fn test_complex_f32_panic() {
        Complex::one().f32(); // Hits the panic line in Complex impl
    }

    #[test]
    #[should_panic(expected = "InvalidOperation")]
    fn test_complex_f64_panic() {
        Complex::one().f64(); // Hits the panic line in Complex impl
    }

    #[test]
    fn test_floating_point_ops() {
        let val: f64 = 4.0;
        assert_eq!(val.sqrt(), 2.0);
        assert_eq!(val.abs(), 4.0);
        assert_eq!(val.max(10.0), 10.0);
        assert_eq!(val.min(1.0), 1.0);
        // Add more ops as needed to hit all trait methods
        let _ = val.sin();
        let _ = val.cos();
        let _ = val.tan();
        let _ = val.tanh();
        let _ = val.exp();
        let _ = val.ln();
        let _ = val.log10();
        let _ = val.round();
    }
}
