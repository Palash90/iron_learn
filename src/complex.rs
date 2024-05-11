//! The `Complex` module provides operations for complex number arithmetic.

use std::ops::{Add, Div, Mul, Sub};

/// A complex number represented by its real and imaginary parts.
#[derive(Debug, PartialEq, Copy, Clone)]
pub struct Complex {
    real: f64,
    imaginary: f64,
}

impl Complex {
    /// Constructs a new `Complex` number.
    ///
    /// # Parameters
    /// - `real`: The real part of the complex number.
    /// - `imaginary`: The imaginary part of the complex number.
    ///
    /// # Returns
    /// A new instance of `Complex`.
    ///
    /// # Example
    ///
    /// ```
    /// use iron_learn::Complex;
    ///
    /// let a = Complex::new(1.0, 2.0);
    /// ```
    pub fn new(real: f64, imaginary: f64) -> Self {
        Self { real, imaginary }
    }
}
impl Add for Complex {
    type Output = Self;

    /// Adds two complex numbers.
    ///
    /// # Parameters
    /// - `rhs`: The right-hand side complex number to add.
    ///
    /// # Returns
    /// The sum of the two complex numbers.
    ///
    /// # Example
    ///
    /// ```
    /// use iron_learn::Complex;
    ///
    /// let a = Complex::new(1.0, 2.0);
    /// let b = Complex::new(3.0, 4.0);
    ///
    /// let r = a + b;
    /// assert_eq!(Complex::new(4.0, 6.0), r);
    /// ```
    ///

    fn add(self, rhs: Self) -> Self {
        Self {
            real: self.real + rhs.real,
            imaginary: self.imaginary + rhs.imaginary,
        }
    }
}

impl Sub for Complex {
    type Output = Self;

    /// Subtracts one complex number from another.
    ///
    /// # Parameters
    /// - `rhs`: The right-hand side complex number to subtract.
    ///
    /// # Returns
    /// The difference of the two complex numbers.
    ///
    /// # Example
    ///
    /// ```
    /// use iron_learn::Complex;
    ///
    /// let a = Complex::new(1.0, 2.0);
    /// let b = Complex::new(3.0, 4.0);
    ///
    /// let r = a - b;
    /// assert_eq!(Complex::new(-2.0, -2.0), r);
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

    /// Multiplies two complex numbers.
    ///
    /// # Parameters
    /// - `rhs`: The right-hand side complex number to multiply.
    ///
    /// # Returns
    /// The product of the two complex numbers.
    ///
    /// # Example
    ///
    /// ```
    /// use iron_learn::Complex;
    ///
    /// let a = Complex::new(1.0, 2.0);
    /// let b = Complex::new(3.0, 4.0);
    ///
    /// let r = a * b;
    /// assert_eq!(Complex::new(-5.0, 10.0), r);
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

    /// Divides one complex number by another.
    ///
    /// # Parameters
    /// - `rhs`: The right-hand side complex number to divide by.
    ///
    /// # Returns
    /// The quotient of the two complex numbers.
    ///
    /// # Example
    ///
    /// ```
    /// use iron_learn::Complex;
    ///
    /// let a = Complex::new(1.0, 2.0);
    /// let b = Complex::new(3.0, 4.0);
    ///
    /// let r = a / b;
    /// assert_eq!(Complex::new(0.44, 0.08), r);
    /// ```
    fn div(self, rhs: Self) -> Self {
        let a = self.real;
        let b = self.imaginary;

        let c = rhs.real;
        let d = rhs.imaginary;

        let divisor = c * c + d * d;

        let real = (a * c + b * d) / divisor;
        let imaginary = (b * c - a * d) / divisor;

        Self {
            real: real,
            imaginary: imaginary,
        }
    }
}
