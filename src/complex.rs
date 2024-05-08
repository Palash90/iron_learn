use std::ops::{Add, Div, Mul, Sub};

#[derive(Debug, PartialEq, Copy, Clone)]
pub struct Complex {
    real: f64,
    imaginary: f64,
}

impl Complex {
    pub fn new(real: f64, imaginary: f64) -> Self {
        Self { real, imaginary }
    }
}
impl Add for Complex {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            real: self.real + rhs.real,
            imaginary: self.imaginary + rhs.imaginary,
        }
    }
}

impl Sub for Complex {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self {
            real: self.real - rhs.real,
            imaginary: self.imaginary - rhs.imaginary,
        }
    }
}

impl Mul for Complex {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self {
            real: ((self.real * rhs.real) - (self.imaginary * rhs.imaginary)),
            imaginary: ((self.imaginary * rhs.real) + (self.real * rhs.imaginary)),
        }
    }
}

impl Div for Complex {
    type Output = Self;

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
