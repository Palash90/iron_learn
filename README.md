# iron_learn
A pure Rust Machine Learning Library

## Status
Version 0.1.0 released with limited Matrix Manipulation abilities. Under active development for further implementation support.

## Overview
This library is designed to facilitate machine learning tasks with a focus on linear algebra operations. Currently, the library supports matrix addition and multiplication, providing a robust foundation for building more complex machine learning algorithms.

## Modules

### `tensor`
At the core of the library, the `tensor` module supports multi-dimensional `Tensor` data structure. It includes methods for tensor instantiation and defines the `+` operator for tensor addition and the `*` operator for tensor multiplication. Additionally, it features a method for the Hadamard product, an element-wise multiplication operation.

The creation, multiplication and addition operation on `Tensor` can fail due to multiple reasons and hence, it returns a result object instead of panic!. The library expects the user to handle the result as per need.

__*N.B:*__ This Module is limited to only two dimensional Matrix Support. This is a temporary restriction and we plan to remove this restriction in future.

### `matrix`
The `matrix` module provides the `Matrix` structure which serves as a wrapper for the `Tensor` object of two dimensions, enabling matrix operations. It defines the `+` and `*` operators for matrix addition and multiplication, respectively, mirroring the behavior of tensors. It also enables `multiply` method to support hadamard product of two matrices.

### `vector`
The `vector` module provides `Vector` type which is a specialized wrapper for the `Tensor` object, tailored for vector operations. It defines the `+` operator for vector addition and the `*` operator for computing the dot product between two vectors.

### `complex`
The `complex` module is experimental and provides a representation of complex numbers, which are fundamental in various machine learning computations, especially in handling operations involving complex-valued data. The `Complex` type supports all arithmatic operations.

### `numeric`
This module defines all supported numeric types necessary for machine learning operations, including integer, unsigned, and floating-point variants, as well as the custom type `Complex`.


## Usage

To use the library, include the following in your project:

```rust
use crate::complex::Complex;
use crate::matrix::Matrix;
use crate::tensor::Tensor;
use crate::vector::Vector;
```

## Examples

Here are some examples of how to use the library for basic operations:

### Matrix Addition
```rust
let a = Matrix::new(vec![2, 2], vec![1, 2, 3, 4]); // Define matrix `a`
let b = Matrix::new(vec![2, 2], vec![5, 6, 7, 8]); // Define matrix `b`
let c = (a + b).unwrap(); // Perform matrix addition
```

### Matrix Multiplication
```rust
let a = Matrix::new(vec![2, 2], vec![1, 2, 3, 4]); // Define matrix `a`
let b = Matrix::new(vec![2, 2], vec![5, 6, 7, 8]); // Define matrix `b`
let c = (a * b).unwrap(); // Perform matrix multiplication
```

### Matrix Hadamard Product
```rust
let a = Matrix::new(vec![2, 2], vec![1, 2, 3, 4]); // Define matrix `a`
let b = Matrix::new(vec![2, 2], vec![5, 6, 7, 8]); // Define matrix `b`
let c = a.multiply(b).unwrap(); // Perform matrix hadamard product
```

### Vector Addition
```rust
let a = Vector::new(vec![2], vec![1, 2]); // Define matrix `a`
let b = Vector::new(vec![2], vec![3, 4]); // Define matrix `b`
let c = (a + b).unwrap(); // Perform matrix addition
```

### Vector Dot Product
```rust
let a = Vector::new(vec![2], vec![1, 2]); // Define matrix `a`
let b = Vector::new(vec![2], vec![3, 4]); // Define matrix `b`
let c = (a * b).unwrap(); // Perform matrix addition
```

### Complex Number Arithmatic
```rust
let a = Complex::new(1.0, 2.0);
let b = Complex::new(3.0, 4.0);

let c = a + b;
let c = a - b;
let c = a * b;
let c = a / b;
```

### Complex Number Matrix Addition & Multiplication
```rust
let a = Complex::new(1.0, 2.0);
let b = Complex::new(3.0, 4.0);
let c = Complex::new(5.0, 6.0);
let d = Complex::new(7.0, 8.0);

let m1 = Matrix::new(vec![2, 2], vec![a, b, c, d]).unwrap();

let m2 = Matrix::new(vec![2, 2], vec![a, c, b, d]).unwrap();

let result = (m1 + m2).unwrap();
let result = (m1 * m2).unwrap();
```    