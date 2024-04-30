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
            let err = String::from(format!("Data length ({}) does not match total num of elements provided by dimensions ({}))", data.len(), size));
            return Err(err);
        }

        Ok(Self {
            dimensions,
            data: data,
        })
    }
}

#[cfg(test)]
#[test]
fn test_new() {
    let t = Tensor::new(vec![1u32, 2u32], vec![1i8, 2i8]).unwrap();

    assert_eq!(t.dimensions, vec![1u32, 2u32]);
    assert_eq!(t.data, vec![1i8, 2i8]);
}
