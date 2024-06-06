# iron_learn
A pure Rust Machine Learning Library

## Status
Version 0.4.0 released with Logistic Regression Algorithm. Now you can directly use `linear_regression` or `logistic_regression` functions instead of directly using `gradient_descent` function, which has been marked as deprecated and is going to introduce breaking changes and in future may be removed. Under active development for further implementation support.

__*N.B:*__ This library was built as a product of Rust Learning Efforts. So, "School Grade" algorithm is used for Matrix Multiplication. I have plans for improving on the performance part though in future.

## Overview
This library is designed to facilitate machine learning tasks with a focus on linear algebra operations. Currently, the library supports matrix addition, subtraction, multiplication, transpose and scaling by a scalar providing a robust foundation for building more complex machine learning algorithms.

## Modules

### `tensor`
At the core of the library, the `tensor` module supports multi-dimensional `Tensor` data structure. It includes methods for tensor instantiation and defines the `+` operator for tensor addition, the `-` operator for subtraction and the `*` operator for tensor multiplication. These operators however, take ownership of the variables(both `rhs` and `lhs`). So, the variables cannot be used later. To facilitate performing these operations without the ownership issue, it also provides `add`, `sub` and `mul` methods to perform addition, subtraction and division by borrowing the variables for operation.

 Additionally, it features a method `t` for transpose and a `multiply` method for the Hadamard product, an element-wise multiplication operation. It also pose `scale` method for scaling by a scalar value.

These operations on `Tensor` can fail due to multiple reasons and hence, it returns a result object. The library assumes a better error handling by the user rather than causing `panic`.

__*N.B:*__ This Module is limited to only two dimensional Matrix Support. This is a temporary restriction and I plan to remove this restriction in future.

### `complex`
The `complex` module is experimental and provides a representation of complex numbers, which are fundamental in various machine learning computations, especially in handling operations involving complex-valued data. The `Complex` type supports all arithmatic operations.

### `numeric`
This module defines all supported numeric types necessary for machine learning operations, including integer, unsigned, and floating-point variants, as well as the custom type `Complex`.

### `gradient_descent`
The `gradient_descent` function performs a single step of the gradient descent optimization algorithm. It can work for both Linear Regression and Logistic Regression, configured by a boolean parameter.

### ~~`matrix`~~ (Use 2-Dimensional `Tensor` instead)
The `matrix` module provides the `Matrix` structure which serves as a wrapper for the `Tensor` object of two dimensions, enabling matrix operations. It defines the `+` and `*` operators for matrix addition and multiplication, respectively, mirroring the behavior of tensors. It also enables `multiply` method to support hadamard product of two matrices.

### ~~`vector`~~ (Use 1-Dimensional `Tensor` instead)
The `vector` module provides `Vector` type which is a specialized wrapper for the `Tensor` object, tailored for vector operations. It defines the `+` operator for vector addition and the `*` operator for computing the dot product between two vectors.

## Usage

To use the library, include the following in your project:

```rust
use iron_learn::Complex;
use iron_learn::Matrix;
use iron_learn::Tensor;
use iron_learn::Vector;
```

## Examples

Here are some examples of how to use the library for basic operations:

### Tensor Addition
```rust
let a = Tensor::new(vec![2, 2], vec![1, 2, 3, 4]).unwrap(); // Define matrix `a`
let b = Tensor::new(vec![2, 2], vec![5, 6, 7, 8]).unwrap(); // Define matrix `b`
let c = (a + b).unwrap(); // Perform matrix addition. Causes a move. Cannot use a or b later.

// No move syntax
let a = Tensor::new(vec![2, 2], vec![1, 2, 3, 4]).unwrap(); // Define matrix `a`
let b = Tensor::new(vec![2, 2], vec![5, 6, 7, 8]).unwrap(); // Define matrix `b`
let c = a.add(&b).unwrap(); // Perform matrix addition without taking ownership..
```

### Tensor Subtraction
```rust
let a = Tensor::new(vec![2, 2], vec![1, 2, 3, 4]).unwrap(); // Define matrix `a`
let b = Tensor::new(vec![2, 2], vec![5, 6, 7, 8]).unwrap(); // Define matrix `b`
let c = (a - b).unwrap(); // Perform matrix addition. Causes a move. Cannot use a or b later.

// No move syntax
let a = Tensor::new(vec![2, 2], vec![1, 2, 3, 4]).unwrap(); // Define matrix `a`
let b = Tensor::new(vec![2, 2], vec![5, 6, 7, 8]).unwrap(); // Define matrix `b`
let c = a.sub(&b).unwrap(); // Perform matrix addition without taking ownership..
```

### Tensor Multiplication
```rust
let a = Tensor::new(vec![2, 2], vec![1, 2, 3, 4]).unwrap(); // Define matrix `a`
let b = Tensor::new(vec![2, 2], vec![5, 6, 7, 8]).unwrap(); // Define matrix `b`
let c = (a * b).unwrap(); // Perform matrix multiplication. Causes a move. Cannot use a or b later.

// No move syntax
let a = Tensor::new(vec![2, 2], vec![1, 2, 3, 4]).unwrap(); // Define matrix `a`
let b = Tensor::new(vec![2, 2], vec![5, 6, 7, 8]).unwrap(); // Define matrix `b`
let c = a.add(&b).unwrap(); // Perform matrix multiplication without taking ownership.
```

### Tensor Hadamard Product
```rust
let a = Tensor::new(vec![2, 2], vec![1, 2, 3, 4]).unwrap(); // Define matrix `a`
let b = Tensor::new(vec![2, 2], vec![5, 6, 7, 8]).unwrap(); // Define matrix `b`
let c = a.multiply(b).unwrap(); // Perform matrix hadamard product
```

### Tensor Transpose
```rust
let m = Tensor::new(vec![6], vec![1, 2, 3, 4, 5, 6]).unwrap();
 m.t().unwrap();
```

### Tensor Scaling
```rust
let m = Tensor::new(vec![6], vec![1, 2, 3, 4, 5, 6]).unwrap();
let r = Tensor::new(vec![6], vec![5, 10, 15, 20, 25, 30]).unwrap();
 assert-eq!(m.scale(5).unwrap(), r);
```

### Gradient Descent Optimization
```rust
use iron_learn::Tensor;
use iron_learn::gradient_descent::gradient_descent;

let learning_rate: f64 = 0.01;
let w = Tensor::new(vec![2, 1], vec![3.0, 4.0]).unwrap();
let x = Tensor::new(vec![1, 2], vec![3.0, 4.0]).unwrap();
let y = Tensor::new(vec![1, 1], vec![5.0]).unwrap();
let w = gradient_descent(&x, &y, &w, learning_rate, true);
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

### Complex Number Tensor Addition & Multiplication
```rust
let a = Complex::new(1.0, 2.0);
let b = Complex::new(3.0, 4.0);
let c = Complex::new(5.0, 6.0);
let d = Complex::new(7.0, 8.0);

let m1 = Tensor::new(vec![2, 2], vec![a, b, c, d]).unwrap();

let m2 = Tensor::new(vec![2, 2], vec![a, c, b, d]).unwrap();

let result = m1.add(&m2).unwrap();
let result = m1.mul(&m2).unwrap();
``` 

### ~~Matrix Addition~~ (Deprecated, use 2-Dimensional `Tensor` instead)
```rust
let a = Matrix::new(vec![2, 2], vec![1, 2, 3, 4]).unwrap(); // Define matrix `a`
let b = Matrix::new(vec![2, 2], vec![5, 6, 7, 8]).unwrap(); // Define matrix `b`
let c = (a + b).unwrap(); // Perform matrix addition
```

### ~~Matrix Multiplication~~ (Deprecated, use 2-Dimensional `Tensor` instead)
```rust
let a = Matrix::new(vec![2, 2], vec![1, 2, 3, 4]).unwrap(); // Define matrix `a`
let b = Matrix::new(vec![2, 2], vec![5, 6, 7, 8]).unwrap(); // Define matrix `b`
let c = (a * b).unwrap(); // Perform matrix multiplication
```

### ~~Matrix Hadamard Product~~ (Deprecated, use 2-Dimensional `Tensor` instead)
```rust
let a = Matrix::new(vec![2, 2], vec![1, 2, 3, 4]).unwrap(); // Define matrix `a`
let b = Matrix::new(vec![2, 2], vec![5, 6, 7, 8]).unwrap(); // Define matrix `b`
let c = a.multiply(b).unwrap(); // Perform matrix hadamard product
```

### ~~Vector Addition~~ (Deprecated, use 1-Dimensional `Tensor` instead)
```rust
let a = Vector::new(vec![2], vec![1, 2]).unwrap(); // Define matrix `a`
let b = Vector::new(vec![2], vec![3, 4]).unwrap(); // Define matrix `b`
let c = (a + b).unwrap(); // Perform matrix addition
```

### ~~Vector Dot Product~~ (Deprecated, use 1-Dimensional `Tensor` instead)
```rust
let a = Vector::new(vec![2], vec![1, 2]).unwrap(); // Define matrix `a`
let b = Vector::new(vec![2], vec![3, 4]).unwrap(); // Define matrix `b`
let c = (a * b).unwrap(); // Perform matrix addition
```

