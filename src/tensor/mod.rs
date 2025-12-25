use crate::Numeric;
pub mod math;

pub trait Tensor<T: Numeric>: Sized {
    fn print_matrix(&self) -> () {
        let data = self.get_data();

        let rows = self.get_shape()[0] as usize;
        let cols = match self.get_shape().len() {
            2 => self.get_shape()[1] as usize,
            _ => 1,
        };

        for r in 0..rows {
            for c in 0..cols {
                print!("{:.6?}\t", data[r * cols + c]);
            }
            println!();
        }
    }

    /* Creation */
    // Returns a Tensor with all zero values
    fn zeroes(shape: &Vec<u32>) -> Self;

    // Returns a Tensor with all one values
    fn ones(shape: &Vec<u32>) -> Self;

    // Creates a new tensor with the provided shape and data
    fn new(shape: Vec<u32>, data: Vec<T>) -> Result<Self, String>;

    /* Retrieval */
    // Returns the shape of the Tensor
    fn get_shape(&self) -> &Vec<u32>;

    // Returns the data
    fn get_data(&self) -> Vec<T>;

    /* Device API */
    // Synchronize with GPU, if running on GPU
    fn synchronize();

    /* Matrix related operations */
    // Adds two Tensors and returns a tensor
    fn add(&self, rhs: &Self) -> Result<Self, String>;

    // Subtracts rhs from current tensor
    fn sub(&self, rhs: &Self) -> Result<Self, String>;

    // Matrix multiplication
    fn mul(&self, rhs: &Self) -> Result<Self, String>;

    // Transpose a 2D matrix, only supported upto 2D
    fn t(&self) -> Result<Self, String>;

    // Hadamard product
    fn multiply(&self, rhs: &Self) -> Result<Self, String>;

    // Element wise scaling
    fn scale(&self, scalar: T) -> Result<Self, String>;

    /* Reducers */
    // Reduces rows column wise
    fn sum(&self) -> Result<Self, String>;
}
