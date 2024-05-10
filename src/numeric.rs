use crate::complex::Complex;

pub trait Numeric:
    Copy
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
{
    fn zero() -> Self;
    fn one() -> Self;
}

// Implement Numeric for all the Numeric Data Types.
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

impl Numeric for Complex {
    fn zero() -> Self {
        Complex::new(0.0, 0.0)
    }

    fn one() -> Self {
        Complex::new(1.0, 0.0)
    }
}
