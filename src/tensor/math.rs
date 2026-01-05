use crate::numeric::FloatingPoint;
use crate::Tensor;

/// Trait providing common element-wise mathematical operations for tensors.
///
/// The implementing tensor type defines `MathOutput` as the concrete tensor
/// result type (for example `CpuTensor<f64>` or `GpuTensor<f32>`). Each
/// operation returns a `Result` so implementations can surface errors from
/// allocation or device/kernel failures.
pub trait TensorMath<T: FloatingPoint>: Tensor<T> + Sized {
    /// The element type of the resulting math tensor (must be floating point).
    type MathOutputElem: FloatingPoint;

    /// The resulting tensor type returned by math operations.
    type MathOutput: Tensor<Self::MathOutputElem>;

    /// Apply the sigmoid function element-wise and return the resulting tensor.
    fn sigmoid(&self) -> Result<Self::MathOutput, String>;

    /// Compute the base-10 logarithm element-wise.
    fn log(&self) -> Result<Self::MathOutput, String>;

    /// Compute the natural logarithm element-wise.
    fn ln(&self) -> Result<Self::MathOutput, String>;

    /// Compute the sine element-wise.
    fn sin(&self) -> Result<Self::MathOutput, String>;

    /// Compute the cosine element-wise.
    fn cos(&self) -> Result<Self::MathOutput, String>;

    /// Compute the tangent element-wise.
    fn tan(&self) -> Result<Self::MathOutput, String>;

    /// Compute the hyperbolic tangent element-wise.
    fn tanh(&self) -> Result<Self::MathOutput, String>;

    /// Compute the exponential (e^x) element-wise.
    fn exp(&self) -> Result<Self::MathOutput, String>;
}
