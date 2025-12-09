 use crate::tensor_commons::TensorOps;
use crate::GpuTensor;
use crate::Numeric;
use crate::cuda_tensor::Zeroable;
use crate::cuda_tensor::OpType;

// This implementation block requires the necessary bounds for GpuTensor methods.
// We'll use the common bound T: Numeric + Zeroable.
impl<T: Numeric + Zeroable> TensorOps<T> for GpuTensor<T>
where
    T: Numeric, // Redundant but harmless, as Numeric is required by Zeroable in your context
{
    // The trait requires &Vec<u32>, but your get_shape returns Vec<u32>.
    // We will change get_shape to return a reference (&Vec<u32>) to match the trait
    // and then implement the trait method using the (corrected) GpuTensor method.
    // NOTE: I'll assume you will adjust your GpuTensor::get_shape to return &Vec<u32>
    // but for now, I'll use the existing public get_shape and clone the result for the trait.
    // A better approach is to change the GpuTensor method signature.

    // Trait required: fn get_shape(&self) -> &Vec<u32>;
    // GpuTensor public method: pub fn get_shape(&self) -> Vec<u32> { self._shape() }
    // Since the trait requires a reference, we must either change the GpuTensor public
    // method or clone the data (less efficient). I will implement it by changing the
    // public get_shape method in the next step.
    fn get_shape(&self) -> &Vec<u32> {
        // Since the shape is already a Vec<u32> in GpuTensor, we can return a reference.
        &self.shape
    }

    fn mul(&self, rhs: &Self) -> Result<Self, String> {
        self._mul(rhs)
    }

    fn sub(&self, rhs: &Self) -> Result<Self, String> {
        self._add(rhs, true) // Calling the internal _add with sub=true
    }

    fn t(&self) -> Result<Self, String> {
        self._t()
    }

    fn scale(&self, scalar: T) -> Result<Self, String> {
        self.element_op(OpType::SCALE, scalar)
    }

    fn sigmoid(&self) -> Result<Self, String> {
        self.element_op(OpType::SIGMOID, T::zero())
    }

    fn get_data(&self) -> Vec<T> {
        self._data()
    }

    fn new(shape: Vec<u32>, data: Vec<T>) -> Result<Self, String> {
        Self::_new(shape, data)
    }
}