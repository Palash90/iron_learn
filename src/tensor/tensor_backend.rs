use crate::numeric::Numeric;

pub trait TensorBackend {
    fn from_data(shape: Vec<u32>, data: Vec<T>) -> Result<Self, String>
    where
        Self: Numeric;
    fn get_data(&self) -> Vec<T>;

    fn get_shape(&self) -> Vec<u32>;

    fn add(&self, rhs: &Self) -> Result<Self, String> where Self: Numeric;
    fn sub(&self, rhs: &Self) -> Result<Self, String> where Self: Numeric;
    fn mul(&self, rhs: &Self) -> Result<Self, String> where Self: Numeric;
    fn hadamard(&self, rhs: &Self) -> Result<Self, String> where Self: Numeric;
    
    // Core unary operations:
    fn t(&self) -> Result<Self, String> where Self: Numeric;
    fn scale(&self, scalar: T) -> Self; 
    fn exp(operand: &Self) -> Tensor<f64>;
}
