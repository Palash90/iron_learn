use crate::GpuTensor;
use crate::numeric::Numeric;
use crate::tensor::Tensor;
use crate::tensor_commons::TensorOps;

impl<T: Numeric + 'static> TensorOps<T> for GpuTensor<T> where GpuTensor<T>: From<GpuTensor<f64>> {
    fn get_shape(&self) -> &Vec<u32> {
        &self.shape
    }

    fn mul(&self, rhs: &Self) -> Result<Self, String> {
        self.mul(rhs)
    }

    fn sub(&self, rhs: &Self) -> Result<Self, String> {
        self.sub(rhs)
    }

    fn t(&self) -> Result<Self, String> {
        self.t()
    }

    fn scale(&self, scalar: T) -> Result<Self, String> {
        Ok(self.scale(scalar))
    }

    fn sigmoid(&self) -> Result<Self, String> {
        if std::any::TypeId::of::<T>() != std::any::TypeId::of::<f64>() {
            return Err("Sigmoid is only implemented for Tensor<f64>".to_string());
        }

        let as_f64_tensor = Tensor::new(
            self.shape.clone(),
            self.data.iter().map(|&x| x.f64()).collect(),
        )
        .unwrap();

        Ok(Tensor::sigmoid(&as_f64_tensor).into())
    }
    
    fn get_data(&self) -> Vec<T> {
        self.get_data()
    }
    
    fn new(shape: Vec<u32>, data: Vec<T>) -> Result<Self, String> {
        Self::new(shape, data)
    }
}
