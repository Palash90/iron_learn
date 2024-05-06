use crate::complex::Complex;

pub trait Numeric:
    Sized
    + Copy
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
{
}

// Implement Numeric for all the Numeric Data Types.
impl Numeric for i8 {}
impl Numeric for i16 {}
impl Numeric for i32 {}
impl Numeric for i64 {}
impl Numeric for i128 {}
impl Numeric for isize {}
impl Numeric for u8 {}
impl Numeric for u16 {}
impl Numeric for u32 {}
impl Numeric for u64 {}
impl Numeric for u128 {}
impl Numeric for usize {}
impl Numeric for f32 {}
impl Numeric for f64 {}

impl Numeric for Complex {}
