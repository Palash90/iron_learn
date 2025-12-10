use crate::cpu_tensor::CpuTensor;
use crate::numeric::Numeric;
use crate::tensor::Tensor;

impl<T: Numeric + 'static> Tensor<T> for CpuTensor<T>
where
    CpuTensor<T>: From<CpuTensor<f64>>,
{
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

        let as_f64_tensor = CpuTensor::new(
            self.shape.clone(),
            self.data.iter().map(|&x| x.f64()).collect(),
        )
        .unwrap();

        Ok(CpuTensor::sigmoid(&as_f64_tensor).into())
    }

    fn get_data(&self) -> Vec<T> {
        self.get_data()
    }

    fn new(shape: Vec<u32>, data: Vec<T>) -> Result<Self, String> {
        Self::new(shape, data)
    }

    fn empty(shape: &Vec<u32>) -> Self {
        let data = match shape.len() {
            2 => vec![T::zero(); (shape[0] * shape[1]) as usize],
            _ => vec![T::zero(); shape[0] as usize],
        };
        Self::new(shape.to_vec(), data).expect("Nothing")
    }

    fn synchronize(&self) {}

    fn add(&self, rhs: &Self) -> Result<Self, String> {
        self.add(rhs)
    }

    fn sum(&self) -> Result<Self, String> {
        Ok(self.sum())
    }

    fn log(&self) -> Result<Self, String> {
        self.log()
    }

    fn ln(&self) -> Result<Self, String> {
        self.ln()
    }

    fn sin(&self) -> Result<Self, String> {
        self.sin()
    }

    fn cos(&self) -> Result<Self, String> {
        self.cos()
    }

    fn tan(&self) -> Result<Self, String> {
        self.tan()
    }

    fn tanh(&self) -> Result<Self, String> {
        self.tanh()
    }

    fn exp(&self) -> Result<Self, String> {
        self.exp()
    }

    fn multiply(&self, rhs: &Self) -> Result<Self, String> {
        self.multiply(rhs)
    }
}
