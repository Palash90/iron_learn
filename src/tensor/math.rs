use crate::Numeric;

// Trait for element-wise mathematical functions
pub trait TensorMath<T: Numeric>: Sized {
    // We update the return type to use an associated type from the Tensor trait.
    // This requires a constraint, so let's move the math trait into the Tensor trait
    // or add a trait for the output tensor type.
    //
    // For simplicity and alignment with your goal, let's make T a requirement for the math trait.

    type MathOutput; // This will be the resulting Tensor type (e.g., CpuTensor<f64>)

    // All math functions now return a Tensor of the MathOutput type
    fn sigmoid(&self) -> Result<Self::MathOutput, String>;
    fn log(&self) -> Result<Self::MathOutput, String>;
    fn ln(&self) -> Result<Self::MathOutput, String>;
    fn sin(&self) -> Result<Self::MathOutput, String>;
    fn cos(&self) -> Result<Self::MathOutput, String>;
    fn tan(&self) -> Result<Self::MathOutput, String>;
    fn tanh(&self) -> Result<Self::MathOutput, String>;
    fn exp(&self) -> Result<Self::MathOutput, String>;
    fn scale(&self, scalar: T) -> Result<Self::MathOutput, String>;
}


// The original Tensor trait, now with a supertrait constraint for math operations
// We enforce that any Tensor that implements this, must be able to perform Math operations.
pub trait Tensor<T: Numeric>: Sized + TensorMath<T> {
    // ... (All existing methods like empty, new, add, sub, mul, etc.)
    
    /* Creation */
    fn empty(shape: &Vec<u32>) -> Self;
    fn new(shape: Vec<u32>, data: Vec<T>) -> Result<Self, String>;

    /* Retrieval */
    fn get_shape(&self) -> &Vec<u32>;
    fn get_data(&self) -> Vec<T>;

    /* Device API */
    fn synchronize(&self);

    /* Matrix related operations */
    fn add(&self, rhs: &Self) -> Result<Self, String>;
    fn sub(&self, rhs: &Self) -> Result<Self, String>;
    fn mul(&self, rhs: &Self) -> Result<Self, String>;
    fn t(&self) -> Result<Self, String>;
    fn multiply(&self, rhs: &Self) -> Result<Self, String>;

    /* Reducers */
    fn sum(&self) -> Result<Self, String>;
}