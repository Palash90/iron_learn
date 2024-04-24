mod numeric;

pub struct Matrix<T: numeric::Numeric> {
    pub dimensions: Vec<u32>,
    pub data: Vec<T>,
}

impl<T: numeric::Numeric> Matrix<T> {
    pub fn new(dimensions: Vec<u32>) -> Self {
        Matrix {
            dimensions,
            data: Vec::new(),
        }
    }
}
