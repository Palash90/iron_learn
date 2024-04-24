pub mod add;
mod numeric;

#[derive(Debug)]
pub struct Tensor<T: numeric::Numeric> {
    pub dimensions: Vec<u32>,
    pub data: Vec<T>,
}

impl<T: numeric::Numeric> Tensor<T> {
    pub fn new(dimensions: Vec<u32>) -> Self {
        let mut size = 1;

        for i in &dimensions {
            size *= i;
        }

        Self {
            dimensions,
            data: Vec::with_capacity(size as usize),
        }
    }
}
