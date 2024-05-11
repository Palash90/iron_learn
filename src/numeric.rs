//! # Numeric Trait Module
//!
//! This module defines the `Numeric` trait which abstracts over the basic arithmetic operations
//! and provides methods to return special values like zero and one. It is implemented for various
//! built-in numeric types and a custom `Complex` type, enabling their use in mathematical contexts
//! where generic operations are required.
//!
//! ## Implementations
//! The `Numeric` trait is implemented for standard integer and floating-point types provided by Rust,
//! as well as for a custom `Complex` type defined in the `complex` module. This allows the types to be
//! used interchangeably in algorithms that require basic arithmetic operations.
//!
//! ## Usage
//! The trait can be used to perform arithmetic operations in a generic way. For example, it can be
//! used to write functions that compute the sum of two numbers, regardless of whether they are integers,
//! floating-point numbers, or complex numbers.
//!
//! ## Features
//! - Generic arithmetic operations: `Add`, `Sub`, `Mul`, `Div`.
//! - Special values: `zero()` and `one()` methods to return zero and one of the implementing type.
//! - Compatibility with custom types: Can be implemented for user-defined types like `Complex`.

use crate::complex::Complex;

/// The `Numeric` trait defines a set of operations that numeric types must support.
/// It includes basic arithmetic operations and the ability to return special values like zero and one.
///
/// Types implementing `Numeric` can be used generically in contexts where arithmetic operations are required.
pub trait Numeric:
    Copy
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
{
    /// Returns the zero value of the type.
    fn zero() -> Self;

    /// Returns the one value of the type.
    fn one() -> Self;
}

// Implementations of the Numeric trait for various built-in numeric types.
impl Numeric for i8 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
}
impl Numeric for i16 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
}
impl Numeric for i32 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
}
impl Numeric for i64 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
}
impl Numeric for i128 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
}
impl Numeric for isize {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
}
impl Numeric for u8 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
}
impl Numeric for u16 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
}
impl Numeric for u32 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
}
impl Numeric for u64 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
}
impl Numeric for u128 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
}
impl Numeric for usize {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
}
impl Numeric for f32 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
}
impl Numeric for f64 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
}

/// Implementation of `Numeric` for the `Complex` type from the `complex` module.
impl Numeric for Complex {
    /// Returns a complex number representing zero.
    fn zero() -> Self {
        Complex::new(0.0, 0.0)
    }

    /// Returns a complex number representing one.
    fn one() -> Self {
        Complex::new(1.0, 0.0)
    }
}
