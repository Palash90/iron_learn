use crate::matrix::Matrix;

impl Matrix {
    pub fn add(self: &Matrix) -> u32 {
        self.row + self.column
    }
}
