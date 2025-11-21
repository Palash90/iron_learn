# iron_learn

A high-performance, pure Rust machine learning library with GPU acceleration support. Designed for linear algebra operations and gradient-based optimization algorithms with an emphasis on numerical stability and computational efficiency.

## Release Status

**Version 0.5.0** - Production Release

- ✅ GPU-accelerated logistic regression with CUDA support
- ✅ Enhanced numerical stability with feature normalization
- ✅ Optimized tensor operations with streaming computation
- ✅ Comprehensive error handling and validation
- ✅ Production-ready with extensive test coverage

Previous releases: v0.4.1 (prediction functions), v0.4.0 (logistic regression), v0.3.0 (matrix operations)

### Hardware Acceleration

This release introduces GPU-accelerated gradient descent through CUDA kernels. The implementation includes:
- Matrix-vector multiplication (gemv) for efficient logistic computation
- Sigmoid activation function on GPU
- Batch gradient computation via X^T · loss operations
- Memory-efficient streaming and synchronization

### Performance Notes

The library uses fundamental matrix multiplication algorithms appropriate for development and educational purposes. Production-scale performance optimization is planned for future releases. GPU acceleration is recommended for large datasets (m > 10,000 samples).

## Overview

The iron_learn library provides a comprehensive toolkit for machine learning applications with a focus on linear algebra and optimization. It features:

- **Tensor Operations**: Multi-dimensional tensor support with element-wise and algebraic operations
- **Linear Algebra**: Complete matrix operations (addition, subtraction, multiplication, transpose, scaling)
- **Gradient Descent**: CPU and GPU-accelerated optimization algorithms
- **Complex Numbers**: Native support for complex-valued computations
- **Type Flexibility**: Generic numeric type system supporting integers, floats, and complex numbers
- **Error Handling**: Comprehensive error propagation with Result-based API

## Architecture & Key Components

## Modules

### `tensor` Module

The foundational module providing multi-dimensional tensor abstractions. Features include:

- **Tensor Structure**: Generic, type-agnostic tensor representation supporting row-major layout
- **Operators**: Overloaded `+` (addition), `-` (subtraction), `*` (multiplication) for ownership-taking operations
- **Borrowing Methods**: Non-consuming `add()`, `sub()`, `mul()` methods for reusable computation
- **Specialized Operations**: Transpose (`t`), Hadamard product (`multiply`), scalar scaling (`scale`)
- **Element-wise Functions**: Exponential (`_exp`), activation functions
- **Data Access**: Methods for shape inspection and data retrieval

**Limitations**: Currently restricted to 2D matrices. N-dimensional tensor support is planned for future releases.

**Usage Pattern**:
```rust
// Consuming operations (take ownership)
let c = (a + b)?;  // a and b are moved

// Borrowing operations (reusable)
let c = a.add(&b)?;  // a and b remain available
```

### `complex` Module

Experimental module providing first-class complex number support:

- **Complex Type**: Immutable, copy-friendly representation with real and imaginary components
- **Arithmetic**: All standard operations (+, -, *, /) for complex arithmetic
- **Integration**: Seamless compatibility with Tensor operations for complex-valued computations
- **Display**: Formatted output for human-readable complex numbers
- **CUDA Ready**: Copy semantics for GPU memory transfers

### `numeric` Module

Type system abstraction layer enabling generic operations across numeric types:

- **Supported Types**: i32, i64, u32, u64, f32, f64, Complex
- **Traits**: Numeric, SignedNumeric for type class abstraction
- **Operations**: Unified interface for arithmetic and mathematical functions
- **Extensibility**: Framework for adding new numeric types

### `gradient_descent` Module

Optimization module providing both CPU and GPU-accelerated gradient-based algorithms:

- **Core Algorithm**: Single-step gradient descent with configurable learning rate
- **Regression Variants**: 
  - Linear regression (MSE loss)
  - Logistic regression (sigmoid activation + binary cross-entropy)
- **Enhancements**:
  - Feature normalization (z-score standardization)
  - Bias term handling
  - Batch processing support
- **GPU Acceleration**: CUDA kernel integration for large-scale optimization

**Note**: The `gradient_descent` function is lower-level; use `linear_regression` or `logistic_regression` for typical workflows.

### `app_context` Module

Application initialization and global state management:

- **Context Management**: OnceLock-based thread-safe global context
- **GPU Detection**: Automatic CUDA availability detection
- **Version Tracking**: Application version and capability flags

### `gpu_regression` Module (New)

GPU-accelerated logistic regression implementation using CUDA:

- **Preprocessing**: Feature normalization and bias handling
- **Kernels**: Matrix operations, sigmoid activation, gradient computation
- **Streaming**: Asynchronous computation with synchronization points
- **Prediction**: GPU-based inference on test data

## Usage

To use the library, include the following in your project:

```rust
use iron_learn::Complex;
use iron_learn::Tensor;
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
