pub mod add;

use crate::numeric::Numeric;

#[derive(Debug, PartialEq)]
pub struct Tensor<T: Numeric> {
    shape: Vec<u32>,
    data: Vec<T>,
}

impl<T: Numeric> Tensor<T> {
    pub fn new(shape: Vec<u32>, data: Vec<T>) -> Result<Self, String> {
        if shape.len() == 0 || shape.len() > 2 {
            return Err(format!(
                "TemporaryShapeRestriction: Currently only accepting tensors upto 1 - 2 dimensions"
            ));
        }

        let mut size = 1;

        for i in &shape {
            size *= i;
        }

        if size != data.len().try_into().unwrap() {
            let err = String::from(format!("DataError: Data length ({}) does not match total num of elements provided by dimensions ({}))", data.len(), size));
            return Err(err);
        }

        Ok(Self { shape, data })
    }
}

#[cfg(test)]
#[test]
fn test_new() {
    let t = Tensor::new(vec![1u32, 2u32], vec![1i8, 2i8]).unwrap();

    assert_eq!(t.shape, vec![1u32, 2u32]);
    assert_eq!(t.data, vec![1i8, 2i8]);
}
