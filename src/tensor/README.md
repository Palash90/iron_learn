# Tensor

Tensor is at the heart of this library.

## Example

### Tensor Addition

```rust
use iron_learn::CpuTensor;
use iron_learn::Tensor;

// Create 2x2 matrices
let a: CpuTensor<f32> = CpuTensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
let b: CpuTensor<f32> = CpuTensor::new(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]).unwrap();

println!("Original A:");
a.print_matrix();

println!("Original B:");
b.print_matrix();

// Add tensors without move
let sum = a.add(&b).unwrap();
println!("Sum:");
sum.print_matrix();
```
