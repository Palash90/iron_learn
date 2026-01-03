//! # Complex Number Module
//!
//! Provides native support for complex number arithmetic and tensor operations.
//!
//! Complex numbers are essential for advanced signal processing, Fourier analysis,
//! and certain machine learning applications. This module offers:
//!
//! - **Complex Type**: Immutable, Copy-friendly complex number representation
//! - **Arithmetic Operations**: Full suite of complex arithmetic (+, -, *, /)
//! - **Tensor Integration**: Use Complex numbers as tensor elements
//! - **GPU Compatibility**: DeviceCopy semantics for CUDA integration
//! - **Formatted Output**: Human-readable complex number display
//!
//! # Mathematical Basis
//!
//! A complex number z = a + bi is represented with:
//! - Real part (a): represents the x-coordinate
//! - Imaginary part (b): represents the y-coordinate
//!
//! All operations follow standard complex arithmetic rules.

use std::ops::{Add, Div, Mul, Sub};

#[cfg(feature = "cuda")]
use cust::DeviceCopy;
use serde::{Deserialize, Serialize};

use std::fmt;

/// A complex number represented by real and imaginary components
///
/// This is a lightweight, copy-friendly type suitable for use in tensors and GPU computations.
/// It implements all standard complex arithmetic operations and integrates seamlessly
/// with the tensor framework.
///
/// # Properties
///
/// - **Copy**: Can be freely copied without performance penalty
/// - **DeviceCopy**: Compatible with CUDA memory transfers
/// - **Comparable**: Implements PartialEq for complex comparison
///
/// # Example
///
/// ```
/// use iron_learn::Complex;
///
/// let z1 = Complex::new(3.0, 4.0);   // 3 + 4i
/// let z2 = Complex::new(1.0, 2.0);   // 1 + 2i
///
/// let sum = z1 + z2;                 // 4 + 6i
/// let product = z1 * z2;             // -5 + 10i
/// ```
#[derive(Debug, PartialEq, Copy, Clone, PartialOrd, Serialize, Deserialize)]
#[cfg_attr(feature = "cuda", derive(DeviceCopy))]
pub struct Complex {
    real: f64,
    imaginary: f64,
}

impl Complex {
    /// Creates a new complex number from real and imaginary parts
    ///
    /// # Arguments
    ///
    /// * `real` - The real component of the complex number
    /// * `imaginary` - The imaginary component of the complex number
    ///
    /// # Returns
    ///
    /// A new Complex instance
    ///
    /// # Example
    ///
    /// ```
    /// use iron_learn::Complex;
    ///
    /// let z = Complex::new(3.0, 4.0);  // Represents 3 + 4i
    /// ```
    pub fn new(real: f64, imaginary: f64) -> Self {
        Self { real, imaginary }
    }
}

impl fmt::Display for Complex {
    /// Formats complex number in standard mathematical notation
    ///
    /// Examples:
    /// - `3 + 4i` for Complex::new(3.0, 4.0)
    /// - `2 - 1i` for Complex::new(2.0, -1.0)
    ///
    /// # Example
    ///
    /// ```
    /// use iron_learn::Complex;
    ///
    /// let z = Complex::new(1.0, 2.0);
    /// println!("{}", z);  // Output: "1 + 2i"
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.imaginary >= 0.0 {
            write!(f, "{} + {}i", self.real, self.imaginary)
        } else {
            write!(f, "{} - {}i", self.real, -self.imaginary)
        }
    }
}

impl Add for Complex {
    type Output = Self;

    /// Adds two complex numbers
    ///
    /// Implements complex addition: (a + bi) + (c + di) = (a + c) + (b + d)i
    ///
    /// # Arguments
    ///
    /// * `rhs` - The complex number to add
    ///
    /// # Returns
    ///
    /// The sum of the two complex numbers
    ///
    /// # Example
    ///
    /// ```
    /// use iron_learn::Complex;
    ///
    /// let a = Complex::new(1.0, 2.0);
    /// let b = Complex::new(3.0, 4.0);
    /// let sum = a + b;
    /// assert_eq!(sum, Complex::new(4.0, 6.0));
    /// ```

    fn add(self, rhs: Self) -> Self {
        Self {
            real: self.real + rhs.real,
            imaginary: self.imaginary + rhs.imaginary,
        }
    }
}

impl Sub for Complex {
    type Output = Self;

    /// Subtracts one complex number from another
    ///
    /// Implements complex subtraction: (a + bi) - (c + di) = (a - c) + (b - d)i
    ///
    /// # Arguments
    ///
    /// * `rhs` - The complex number to subtract
    ///
    /// # Returns
    ///
    /// The difference of the two complex numbers
    ///
    /// # Example
    ///
    /// ```
    /// use iron_learn::Complex;
    ///
    /// let a = Complex::new(1.0, 2.0);
    /// let b = Complex::new(3.0, 4.0);
    /// let diff = a - b;
    /// assert_eq!(diff, Complex::new(-2.0, -2.0));
    /// ```
    fn sub(self, rhs: Self) -> Self {
        Self {
            real: self.real - rhs.real,
            imaginary: self.imaginary - rhs.imaginary,
        }
    }
}

impl Mul for Complex {
    type Output = Self;

    /// Multiplies two complex numbers
    ///
    /// Implements complex multiplication using the distributive property:
    /// (a + bi)(c + di) = (ac - bd) + (ad + bc)i
    ///
    /// # Arguments
    ///
    /// * `rhs` - The complex number to multiply by
    ///
    /// # Returns
    ///
    /// The product of the two complex numbers
    ///
    /// # Example
    ///
    /// ```
    /// use iron_learn::Complex;
    ///
    /// let a = Complex::new(1.0, 2.0);
    /// let b = Complex::new(3.0, 4.0);
    /// let product = a * b;
    /// assert_eq!(product, Complex::new(-5.0, 10.0));
    /// ```
    fn mul(self, rhs: Self) -> Self {
        Self {
            real: ((self.real * rhs.real) - (self.imaginary * rhs.imaginary)),
            imaginary: ((self.imaginary * rhs.real) + (self.real * rhs.imaginary)),
        }
    }
}

impl Div for Complex {
    type Output = Self;

    /// Divides one complex number by another
    ///
    /// Implements complex division using conjugate multiplication:
    /// (a + bi)/(c + di) = ((ac + bd) + (bc - ad)i) / (c² + d²)
    ///
    /// # Arguments
    ///
    /// * `rhs` - The complex number to divide by
    ///
    /// # Returns
    ///
    /// The quotient of the two complex numbers
    ///
    /// # Panics
    ///
    /// Does not panic on division by zero; returns NaN/Inf values
    ///
    /// # Example
    ///
    /// ```
    /// use iron_learn::Complex;
    ///
    /// let a = Complex::new(1.0, 2.0);
    /// let b = Complex::new(3.0, 4.0);
    /// let quotient = a / b;
    /// assert_eq!(quotient, Complex::new(0.44, 0.08));
    /// ```
    fn div(self, rhs: Self) -> Self {
        let a = self.real;
        let b = self.imaginary;

        let c = rhs.real;
        let d = rhs.imaginary;

        let divisor = c * c + d * d;

        let real = (a * c + b * d) / divisor;
        let imaginary = (b * c - a * d) / divisor;

        Self { real, imaginary }
    }
}

impl std::ops::Neg for Complex {
    type Output = Self;

    /// Negates a complex number
    ///
    /// Multiplies both real and imaginary parts by -1
    ///
    /// # Returns
    ///
    /// The negated complex number
    ///
    /// # Example
    ///
    /// ```
    /// use iron_learn::Complex;
    ///
    /// let a = Complex::new(1.0, 2.0);
    /// let negated = -a;
    /// assert_eq!(negated, Complex::new(-1.0, -2.0));
    /// ```
    fn neg(self) -> Self::Output {
        Complex::new(self.real * -1.0, self.imaginary * -1.0)
    }
}
