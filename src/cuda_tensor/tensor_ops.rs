use crate::cuda_tensor::OpType;
use crate::cuda_tensor::Zeroable;
use crate::tensor_commons::TensorOps;
use crate::GpuTensor;
use crate::Numeric;
use crate::GLOBAL_CONTEXT;

impl<T: Numeric + Zeroable> TensorOps<T> for GpuTensor<T> {
    fn get_shape(&self) -> &Vec<u32> {
        &self.shape
    }

    fn mul(&self, rhs: &Self) -> Result<Self, String> {
        self._mul(rhs)
    }

    fn sub(&self, rhs: &Self) -> Result<Self, String> {
        self._add(rhs, true)
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

    fn synchronize(&self) {
        &(GLOBAL_CONTEXT
            .get()
            .expect("No Context Intialized")
            .stream.as_ref()
            .expect("Stream could not be found"))
        .synchronize();
    }
}
