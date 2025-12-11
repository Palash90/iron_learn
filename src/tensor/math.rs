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
}
