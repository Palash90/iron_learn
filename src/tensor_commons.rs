use crate::Numeric;

pub trait TensorOps<T: Numeric>: Sized {
    // Methods required for general tensor algebra (borrowing)
    fn get_shape(&self) -> &Vec<u32>;
    fn mul(&self, rhs: &Self) -> Result<Self, String>;
    fn sub(&self, rhs: &Self) -> Result<Self, String>;
    fn t(&self) -> Result<Self, String>;
    fn scale(&self, scalar: T) -> Result<Self, String>;

    fn sigmoid(&self) -> Result<Self, String>;

    fn get_data(&self) -> Vec<T>;
    fn new(shape: Vec<u32>, data: Vec<T>) -> Result<Self, String>;
}
