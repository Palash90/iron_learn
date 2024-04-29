pub mod add;
mod numeric;

#[derive(Debug, PartialEq)]
pub struct Tensor<T: numeric::Numeric> {
    pub dimensions: Vec<u32>,
    data: Vec<T>,
}

impl<T: numeric::Numeric> Tensor<T> {
    pub fn new(dimensions: Vec<u32>, data: Vec<T>) -> Result<Self, String> {
        let mut size = 1;

        for i in &dimensions {
            size *= i;
        }

        if size != data.len().try_into().unwrap() {
            return Err(format!("Data length ({}) does not match total num of elements provided by dimensions ({}))", data.len(), size));
        }

        Ok(Self {
            dimensions,
            data: data,
        })
    }
}
