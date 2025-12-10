use crate::cpu_tensor::CpuTensor;
use crate::numeric::Numeric;
use std::fmt;

impl<T: Numeric + fmt::Display> fmt::Display for CpuTensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.display_tensor(0, &self.shape, &self.data, self.shape.len(), f)
    }
}

impl<T: Numeric + fmt::Display> CpuTensor<T> {
    // Recursive function to display the tensor
    fn display_tensor(
        &self,
        index: usize,
        shape: &[u32],
        data: &[T],
        dim: usize,
        f: &mut fmt::Formatter,
    ) -> fmt::Result {
        if shape.is_empty() {
            return write!(f, "{:.2}\t", data[index]);
        }

        let dim_size = shape[0] as usize;
        if dim > 0 {
            write!(f, " ")?;

            for i in 0..dim_size {
                if i > 0 {
                    write!(f, " ")?;
                }
                self.display_tensor(
                    index + i * self.stride(shape),
                    &shape[1..],
                    data,
                    dim - 1,
                    f,
                )?;
            }

            write!(f, "\n")?;
            if dim == self.shape.len() {
                writeln!(f)?; // Newline for the outermost dimension
            }
        }
        Ok(())
    }

    // Helper function to calculate the stride
    fn stride(&self, shape: &[u32]) -> usize {
        shape[1..].iter().product::<u32>() as usize
    }
}
