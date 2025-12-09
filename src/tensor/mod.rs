use crate::Numeric;

pub trait Tensor<T: Numeric>: Sized {
    fn new(shape: Vec<u32>, data: Vec<T>) -> Result<Self, String>;
    fn get_shape(&self) -> &Vec<u32>;
    fn get_data(&self) -> Vec<T>;
    fn synchronize(&self);

    fn add(&self, rhs: &Self) -> Result<Self, String>;
    fn sum(&self) -> Result<Self, String>;
    fn sub(&self, rhs: &Self) -> Result<Self, String>;
    fn mul(&self, rhs: &Self) -> Result<Self, String>;
    fn t(&self) -> Result<Self, String>;
    fn sigmoid(&self) -> Result<Self, String>;
    fn log(&self) -> Result<Self, String>;
    fn ln(&self) -> Result<Self, String>;
    fn sin(&self) -> Result<Self, String>;
    fn cos(&self) -> Result<Self, String>;
    fn tan(&self) -> Result<Self, String>;
    fn tanh(&self) -> Result<Self, String>;
    fn exp(&self) -> Result<Self, String>;
    fn multiply(&self, rhs: &Self) -> Result<Self, String>;
    fn scale(&self, scalar: T) -> Result<Self, String>;
}
