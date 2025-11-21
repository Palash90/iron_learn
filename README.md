# Iron Learn

A pure Rust machine learning library with GPU-accelerated optimization. Built for efficient tensor operations, gradient-based algorithms, and numerical computing with an emphasis on type safety and correctness.

## Features

- **GPU-Accelerated Training**: CUDA kernels for logistic regression on large datasets
- **Comprehensive Tensor Support**: Multi-dimensional arrays with generic numeric types
- **Optimization Algorithms**: Linear and logistic regression with automatic preprocessing
- **Complex Number Arithmetic**: Native support for complex-valued computations
- **Type-Safe API**: Result-based error handling without panics
- **Zero-Copy Operations**: Borrowing methods for efficient computation reuse

## Quick Start

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
iron_learn = "0.5"
```

### Basic Usage

```rust
use iron_learn::Tensor;

// Create 2x2 matrices
let a = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])?;
let b = Tensor::new(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0])?;

// Add tensors without move
let sum = a.add(&b)?;

// Multiply tensors
let product = a.mul(&b)?;
```

## Core Modules

### Tensor

The foundation of the library, providing N-dimensional array operations.

**Key Features:**
- Generic numeric type support (f64, f32, i32, i64, Complex, etc.)
- Row-major storage layout
- Both consuming and borrowing operation variants
- Error handling via `Result` types

**Example:**
```rust
use iron_learn::Tensor;

let x = Tensor::new(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
let y = x.t()?;  // Transpose
let scaled = x.scale(2.0)?;
```

### Gradient Descent

Optimization algorithms for machine learning with automatic preprocessing.

**Features:**
- Z-score feature normalization
- Automatic bias term handling
- Linear regression (MSE loss)
- Logistic regression (cross-entropy loss with sigmoid)
- Support for custom learning rates

**Example:**
```rust
use iron_learn::{Tensor, linear_regression};

let x = Tensor::new(vec![100, 5], x_data)?;
let y = Tensor::new(vec![100, 1], y_data)?;
let w = Tensor::new(vec![6, 1], vec![0.0; 6])?;

for _ in 0..1000 {
    w = linear_regression(&x, &y, &w, 0.01)?;
}
```

### GPU Regression

CUDA-accelerated logistic regression for high-performance training.

**Features:**
- Efficient matrix-vector multiplication on GPU
- Sigmoid activation on device
- Batch gradient computation
- Asynchronous kernel execution with synchronization

**Requirements:**
- CUDA Toolkit installed
- Compatible NVIDIA GPU

**Usage:**
```bash
cargo run -- gpu
```

### Complex Numbers

First-class support for complex-valued computations.

**Example:**
```rust
use iron_learn::Complex;

let z1 = Complex::new(3.0, 4.0);    // 3 + 4i
let z2 = Complex::new(1.0, 2.0);    // 1 + 2i
let product = z1 * z2;              // -5 + 10i
```

Complex numbers integrate seamlessly with tensors:

```rust
let m = Tensor::new(vec![2, 2], vec![
    Complex::new(1.0, 2.0),
    Complex::new(3.0, 4.0),
    Complex::new(5.0, 6.0),
    Complex::new(7.0, 8.0),
])?;

let result = m.add(&m)?;
```

## API Reference

### Tensor Operations

| Operation | Method | Notes |
|-----------|--------|-------|
| Addition | `a.add(&b)` | Borrowing variant |
| Addition | `a + b` | Consuming variant (move semantics) |
| Subtraction | `a.sub(&b)` | Borrowing variant |
| Multiplication | `a.mul(&b)` | Matrix multiplication |
| Hadamard Product | `a.multiply(&b)` | Element-wise multiplication |
| Transpose | `a.t()` | 2D only |
| Scaling | `a.scale(2.0)` | Scalar multiplication |
| Data Access | `a.get_data()` | Get underlying vector |
| Shape Access | `a.get_shape()` | Get dimension vector |

### Regression Functions

| Function | Purpose |
|----------|---------|
| `linear_regression()` | Single optimization step for linear regression |
| `logistic_regression()` | Single optimization step for classification |
| `predict_linear()` | Generate continuous predictions |
| `predict_logistic()` | Generate binary predictions (0 or 1) |

## Architecture

```
iron_learn/
├── tensor       → Multi-dimensional arrays and operations
├── numeric      → Type system for generic computation
├── complex      → Complex number arithmetic
├── gradient_descent → CPU optimization algorithms
├── gpu_regression   → CUDA-accelerated training
├── regression   → High-level training interface
└── app_context  → Global state management
```

## Performance Characteristics

- **Small datasets** (< 1,000 samples): CPU mode is sufficient
- **Medium datasets** (1,000 - 10,000 samples): CPU or GPU
- **Large datasets** (> 10,000 samples): GPU recommended

## Limitations

- Tensors currently limited to 2D matrices (N-dimensional support planned)
- Matrix multiplication uses O(n³) algorithm (suitable for educational use)
- GPU support requires CUDA-capable hardware

## Examples

### Linear Regression
```rust
use iron_learn::{Tensor, linear_regression};

let x_train = Tensor::new(vec![50, 3], training_data)?;
let y_train = Tensor::new(vec![50, 1], training_labels)?;
let mut w = Tensor::new(vec![4, 1], vec![0.0; 4])?;

for i in 0..10000 {
    w = linear_regression(&x_train, &y_train, &w, 0.01)?;
    if i % 1000 == 0 {
        println!("Epoch {}", i);
    }
}

// Predictions
let x_test = Tensor::new(vec![10, 3], test_data)?;
let predictions = predict_linear(&x_test, &w)?;
```

### Logistic Regression (Classification)
```rust
use iron_learn::{Tensor, logistic_regression};

let x_train = Tensor::new(vec![100, 4], x_data)?;
let y_train = Tensor::new(vec![100, 1], y_data)?;
let mut w = Tensor::new(vec![5, 1], vec![0.0; 5])?;

for _ in 0..5000 {
    w = logistic_regression(&x_train, &y_train, &w, 0.01)?;
}

let x_test = Tensor::new(vec![20, 4], test_data)?;
let predictions = predict_logistic(&x_test, &w)?;
```

### Complex Tensor Operations
```rust
use iron_learn::{Tensor, Complex};

let a = Complex::new(1.0, 2.0);
let b = Complex::new(3.0, 4.0);

let m1 = Tensor::new(vec![2, 2], vec![a, b, b, a])?;
let m2 = Tensor::new(vec![2, 2], vec![a, a, b, b])?;

let sum = m1.add(&m2)?;
let product = m1.mul(&m2)?;
```

## Building from Source

```bash
git clone https://github.com/Palash90/iron_learn.git
cd iron_learn
cargo build --release
```

### With CUDA Support

```bash
# Requires CUDA Toolkit installation
cargo build --release
```

## Testing

```bash
cargo test
cargo test --release
```

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please ensure:
- Code passes `cargo test`
- Documentation is updated
- Examples remain functional
- No clippy warnings

## Roadmap
- [ ] CPU SIMD
- [ ] N-dimensional tensor support
- [ ] Distributed training across multiple GPUs
- [ ] Additional optimization algorithms (Adam, RMSprop)
- [ ] Neural network module
- [ ] Performance optimizations for matrix multiplication
- [ ] WASM support

---

**Version**: 0.5.0  
**Author**: Palash Kanti Kundu  
**Repository**: https://github.com/Palash90/iron_learn