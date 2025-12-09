use super::Tensor;
use crate::numeric::Numeric;

// This is the actual implementation of all the operations. This is here to avoid the documentation comment clutter.
impl<T: Numeric> Tensor<T> {
    pub fn _exp(operand: &Self) -> Tensor<f64> {
        let result = operand.data.iter().map(|t| f64::exp(t.f64())).collect();
        Tensor::new(operand.shape.clone(), result).unwrap()
    }

    pub fn _sin(operand: &Self) -> Tensor<f64> {
        let result = operand.data.iter().map(|t| f64::sin(t.f64())).collect();
        Tensor::new(operand.shape.clone(), result).unwrap()
    }

    pub fn _t(&self) -> Result<Self, String> {
        if self.shape.len() > 2 {
            return Err("Only upto 2D tensors can be transposed.".to_string());
        }

        if self.shape.len() == 1 {
            return Ok(self.clone());
        }

        let new_shape = vec![self.shape[1], self.shape[0]];
        let mut new_data = vec![T::zero(); self.data.len()];
        let (rows, cols) = (self.shape[0] as usize, self.shape[1] as usize);
        for i in 0..rows {
            for j in 0..cols {
                new_data[j * rows + i] = self.data[i * cols + j];
            }
        }
        Ok(Self {
            shape: new_shape,
            data: new_data,
        })
    }
    pub fn _data(&self) -> Vec<T> {
        self.data.clone()
    }

    pub fn _shape(&self) -> Vec<u32> {
        self.shape.clone()
    }

    pub fn _add(&self, rhs: &Self, sub: bool) -> Result<Self, String> {
        let mut result = Vec::with_capacity(self.data.len());

        if self.shape != rhs.shape {
            return Err(format!("ShapeMismatch:The dimensions of two matrices are not compatible for addition/subtraction- {:?} {:?}", self.shape, rhs.shape));
        }

        for i in 0..self.data.len() {
            if sub {
                result.push(self.data[i] - rhs.data[i])
            } else {
                result.push(self.data[i] + rhs.data[i])
            }
        }

        Ok(Self {
            shape: self.shape.clone(),
            data: result,
        })
    }

    pub fn check_shape(shape: &[u32]) -> Option<Result<Tensor<T>, String>> {
        if shape.is_empty() {
            return Some(Err(
                "ShapeError: Tensor must have at least one dimension.".to_string()
            ));
        }

        if shape.len() > 2 {
            return Some(Err(
                "TemporaryShapeRestriction: Currently only accepting tensors upto 2 dimensions"
                    .to_string(),
            ));
        }
        None
    }

    pub fn calculate_length(shape: &Vec<u32>) -> u32 {
        let mut size = 1;

        for i in shape {
            size *= i;
        }
        size
    }

    pub fn _new(shape: Vec<u32>, data: Vec<T>) -> Result<Self, String> {
        if let Some(value) = Self::check_shape(&shape) {
            return value;
        }

        let size = Self::calculate_length(&shape);

        if size != (data.len() as u32) {
            let err = format!("DataError: Data length ({}) does not match total num of elements provided by dimensions ({}))", data.len(), size);
            return Err(err);
        }

        Ok(Self { shape, data })
    }

    pub fn hadamard(&self, rhs: &Self) -> Result<Self, String> {
        let mut result = Vec::with_capacity(self.data.len());

        if self.shape != rhs.shape {
            return Err(format!(
                "ShapeMismatch: Mismatch in shape of two Tensors {:?} {:?}",
                self.shape, rhs.shape
            ));
        }

        for i in 0..self.data.len() {
            result.push(self.data[i] * rhs.data[i])
        }

        Ok(Self {
            shape: self.shape.clone(),
            data: result,
        })
    }

    fn _cpu_mul(&self, rhs: &Self) -> Result<Self, String> {
        let rows = self.shape[0] as usize;
        let cols = rhs.shape[1] as usize;
        let common_dim = self.shape[1] as usize;

        let mut data = vec![T::zero(); rows * cols];

        for i in 0..rows {
            for j in 0..cols {
                for k in 0..common_dim {
                    let val = self.data[i * common_dim + k] * rhs.data[k * cols + j];
                    data[i * cols + j] = data[i * cols + j] + val;
                }
            }
        }

        Ok(Tensor {
            shape: vec![rows as u32, cols as u32],
            data,
        })
    }

    pub fn _mul(&self, rhs: &Self) -> Result<Self, String> {
        if self.shape[1] != rhs.shape[0] {
            let s = format!(
                "ShapeMismatch:The dimensions of two matrices are not compatible for multiplication- {:?} {:?}",
                self.shape, rhs.shape
            );
            return Err(s);
        }

        self._cpu_mul(rhs)
    }

    pub fn _s(&self, scalar: T) -> Self {
        let mut new_data = Vec::<T>::new();

        for i in self._data() {
            new_data.push(i * scalar);
        }

        Self {
            shape: self.shape.clone(),
            data: new_data,
        }
    }
}
