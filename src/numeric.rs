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

use serde::{Deserialize, Serialize};

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
    + std::fmt::Display
    + std::fmt::Debug
    + std::cmp::PartialEq
    + std::cmp::PartialOrd
    + for<'de> Deserialize<'de>
    + Serialize
{
    /// Returns the zero value of the type.
    fn zero() -> Self;

    /// Returns the one value of the type.
    fn one() -> Self;

    /// Returns f64 equivalent of t
    fn f64(&self) -> f64;

    /// Returns f32 equivalent of t
    fn f32(&self) -> f32;

    /// Casts the value to Self and returns
    fn from_u32(value: u32) -> Self;

    /// Casts the value to Self and returns
    fn from_f64(value: f64) -> Self;
}

/// The `SignedNumeric` defines all the `Numeric` types that can be signed like `i32`, `i64` etc.
pub trait SignedNumeric: Numeric + std::ops::Neg<Output = Self> {}

// Implementations of the Numeric trait for various built-in numeric types.

impl Numeric for i8 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
    fn f64(&self) -> f64 {
        *self as f64
    }
    fn f32(&self) -> f32 {
        *self as f32
    }
    fn from_u32(value: u32) -> Self {
        value as i8
    }
    fn from_f64(value: f64) -> Self {
        value as i8
    }
}
impl Numeric for i16 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
    fn f64(&self) -> f64 {
        *self as f64
    }
    fn f32(&self) -> f32 {
        *self as f32
    }
    fn from_u32(value: u32) -> Self {
        value as i16
    }
    fn from_f64(value: f64) -> Self {
        value as i16
    }
}
impl Numeric for i32 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
    fn f64(&self) -> f64 {
        *self as f64
    }
    fn f32(&self) -> f32 {
        *self as f32
    }
    fn from_u32(value: u32) -> Self {
        value.try_into().unwrap()
    }
    fn from_f64(value: f64) -> Self {
        value as i32
    }
}
impl Numeric for i64 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
    fn f64(&self) -> f64 {
        *self as f64
    }
    fn f32(&self) -> f32 {
        *self as f32
    }
    fn from_u32(value: u32) -> Self {
        value as i64
    }
    fn from_f64(value: f64) -> Self {
        value as i64
    }
}
impl Numeric for i128 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
    fn f64(&self) -> f64 {
        *self as f64
    }
    fn f32(&self) -> f32 {
        *self as f32
    }
    fn from_u32(value: u32) -> Self {
        value as i128
    }
    fn from_f64(value: f64) -> Self {
        value as i128
    }
}
impl Numeric for isize {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
    fn f64(&self) -> f64 {
        *self as f64
    }
    fn f32(&self) -> f32 {
        *self as f32
    }
    fn from_u32(value: u32) -> Self {
        value as isize
    }
    fn from_f64(value: f64) -> Self {
        value as isize
    }
}

impl Numeric for u8 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
    fn f64(&self) -> f64 {
        *self as f64
    }
    fn f32(&self) -> f32 {
        *self as f32
    }
    fn from_u32(value: u32) -> Self {
        (value).try_into().unwrap()
    }
    fn from_f64(value: f64) -> Self {
        value as u8
    }
}

impl Numeric for u16 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
    fn f64(&self) -> f64 {
        *self as f64
    }
    fn f32(&self) -> f32 {
        *self as f32
    }
    fn from_u32(value: u32) -> Self {
        value as u16
    }
    fn from_f64(value: f64) -> Self {
        value as u16
    }
}
impl Numeric for u32 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
    fn f64(&self) -> f64 {
        *self as f64
    }
    fn f32(&self) -> f32 {
        *self as f32
    }
    fn from_u32(value: u32) -> Self {
        value
    }
    fn from_f64(value: f64) -> Self {
        value as u32
    }
}
impl Numeric for u64 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
    fn f64(&self) -> f64 {
        *self as f64
    }
    fn f32(&self) -> f32 {
        *self as f32
    }
    fn from_u32(value: u32) -> Self {
        value as u64
    }
    fn from_f64(value: f64) -> Self {
        value as u64
    }
}
impl Numeric for u128 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
    fn f64(&self) -> f64 {
        *self as f64
    }
    fn f32(&self) -> f32 {
        *self as f32
    }
    fn from_u32(value: u32) -> Self {
        value as u128
    }
    fn from_f64(value: f64) -> Self {
        value as u128
    }
}
impl Numeric for usize {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
    fn f64(&self) -> f64 {
        *self as f64
    }
    fn f32(&self) -> f32 {
        *self as f32
    }
    fn from_u32(value: u32) -> Self {
        value as usize
    }
    fn from_f64(value: f64) -> Self {
        value as usize
    }
}

impl Numeric for f32 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn f64(&self) -> f64 {
        *self as f64
    }
    fn f32(&self) -> f32 {
        *self
    }
    fn from_u32(value: u32) -> Self {
        value as f32
    }
    fn from_f64(value: f64) -> Self {
        value as f32
    }
}
impl Numeric for f64 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn f64(&self) -> f64 {
        *self
    }
    fn f32(&self) -> f32 {
        *self as f32
    }
    fn from_u32(value: u32) -> Self {
        value as f64
    }
    fn from_f64(value: f64) -> Self {
        value
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

    fn f32(&self) -> f32 {
        panic!("InvalidOperation: Trying to convert a complex number to floating point number");
    }

    fn f64(&self) -> f64 {
        panic!("InvalidOperation: Trying to convert a complex number to floating point number");
    }

    fn from_u32(value: u32) -> Self {
        Complex::new(value as f64, 0.0)
    }
    fn from_f64(value: f64) -> Self {
        Complex::new(value, 0.0)
    }
}

impl SignedNumeric for i8 {}
impl SignedNumeric for i16 {}
impl SignedNumeric for i32 {}
impl SignedNumeric for i64 {}
impl SignedNumeric for i128 {}
impl SignedNumeric for isize {}
impl SignedNumeric for f64 {}
impl SignedNumeric for f32 {}
impl SignedNumeric for Complex {}

/// The `FloatingPoint` defines all the `SignedNumeric` types that can be signed like `f32` and `f64`.
pub trait FloatingPoint: SignedNumeric {
    fn round(&self) -> Self;
    fn sqrt(&self) -> Self;
    fn abs(&self) -> Self;
    fn cos(&self) -> Self;
    fn sin(&self) -> Self;
    fn tan(&self) -> Self;
    fn tanh(&self) -> Self;
    fn exp(&self) -> Self;
    fn ln(&self) -> Self;
    fn log10(&self) -> Self;
    fn max(&self, other: &Self) -> Self {
        if self > other {
            *self
        } else {
            *other
        }
    }
    fn min(&self, other: &Self) -> Self {
        if self < other {
            *self
        } else {
            *other
        }
    }
}

impl FloatingPoint for f32 {
    fn sqrt(&self) -> Self {
        f32::sqrt(*self)
    }
    fn abs(&self) -> Self {
        f32::abs(*self)
    }
    fn cos(&self) -> Self {
        f32::cos(*self)
    }

    fn sin(&self) -> Self {
        f32::sin(*self)
    }

    fn tan(&self) -> Self {
        f32::tan(*self)
    }

    fn tanh(&self) -> Self {
        f32::tanh(*self)
    }

    fn exp(&self) -> Self {
        f32::exp(*self)
    }

    fn ln(&self) -> Self {
        f32::ln(*self)
    }

    fn log10(&self) -> Self {
        f32::log10(*self)
    }

    fn round(&self) -> Self {
        f32::round(*self)
    }
}

impl FloatingPoint for f64 {
    fn sqrt(&self) -> Self {
        f64::sqrt(*self)
    }
    fn abs(&self) -> Self {
        f64::abs(*self)
    }
    fn cos(&self) -> Self {
        f64::cos(*self)
    }

    fn sin(&self) -> Self {
        f64::sin(*self)
    }

    fn tan(&self) -> Self {
        f64::tan(*self)
    }

    fn tanh(&self) -> Self {
        f64::tanh(*self)
    }

    fn exp(&self) -> Self {
        f64::exp(*self)
    }

    fn ln(&self) -> Self {
        f64::ln(*self)
    }

    fn log10(&self) -> Self {
        f64::log10(*self)
    }

    fn round(&self) -> Self {
        f64::round(*self)
    }
}
