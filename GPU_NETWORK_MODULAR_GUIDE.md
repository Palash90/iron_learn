# GPU Neural Network Modular Architecture

## Overview

The Rust GPU neural network code is now organized, with separate modules for layers, activation functions, network building, and training.

## Module Structure

```
src/gpu_regression/
├── mod.rs                      # Module exports
├── gpu_regression_helpers.rs   # Low-level kernel wrappers
├── layers.rs                   # Layer definitions (Linear, Activation)
├── activations.rs              # Activation function definitions
├── builder.rs                  # Network builder and presets
└── trainer.rs                  # Training harness and configuration
```

## Quick Start: Building and Training Networks

### 1. Building a Simple 2-Layer Network

```rust
use crate::gpu_regression::builder::build_simple_2layer;
use cust::module::Module;
use cust::stream::{Stream, StreamFlags};

// Load CUDA kernel module
let ptx = include_str!("../kernels/gpu_kernels.ptx");
let module = Module::from_ptx(ptx, &[])?;
let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

// Build network: Input (52) -> Hidden (64) -> Output (1)
let network = build_simple_2layer(52, 64, 1)?;
```

### 2. Creating a Custom Network

```rust
use crate::gpu_regression::builder::GpuNetworkBuilder;
use crate::gpu_regression::activations::{relu_activation, relu_derivative};

let network = GpuNetworkBuilder::new()
    .add_linear("CustomInput", 52, 128)?           // Input to hidden1
    .add_activation("ReLU1", relu_activation, relu_derivative)
    .add_linear("Hidden2", 128, 64)?               // Hidden1 to hidden2
    .add_activation("ReLU2", relu_activation, relu_derivative)
    .add_linear("CustomOutput", 64, 1)?            // Hidden2 to output
    .build();
```

### 3. Training the Network

```rust
use crate::gpu_regression::trainer::{GpuNetworkTrainer, TrainerConfig};

let config = TrainerConfig {
    learning_rate: 0.001,
    epochs: 10000,
    checkpoint_interval: 1000,
    hidden_size: 64,
};

let mut trainer = GpuNetworkTrainer::new(network, config, module, stream);

let (duration, final_loss) = trainer.fit(
    &x_train,   // Preprocessed training features
    &y_train,   // Training targets
    rows,
    input_cols  // includes bias column
)?;

println!("Training completed in {:?}, final loss: {}", duration, final_loss);
```

### 4. Evaluation

```rust
let test_mse = trainer.evaluate(&x_test, &y_test, test_rows, input_cols)?;
println!("Test MSE: {}", test_mse);
```

## Modifying Network Architecture

### Example: Adding a 3-Layer Network

**Step 1: Define it in builder.rs**

```rust
pub fn build_3layer_custom(
    input_size: usize,
    hidden1: usize,
    hidden2: usize,
    output_size: usize,
) -> CudaResult<GpuNetwork> {
    use super::activations::{relu_activation, relu_derivative};

    let network = GpuNetworkBuilder::new()
        .add_linear("Input_Hidden1", input_size, hidden1)?
        .add_activation("ReLU1", relu_activation, relu_derivative)
        .add_linear("Hidden1_Hidden2", hidden1, hidden2)?
        .add_activation("ReLU2", relu_activation, relu_derivative)
        .add_linear("Hidden2_Output", hidden2, output_size)?
        .build();
    
    Ok(network)
}
```

**Step 2: Use it in your code**

```rust
let network = build_3layer_custom(52, 128, 64, 1)?;
let mut trainer = GpuNetworkTrainer::new(network, config, module, stream);
trainer.fit(&x_train, &y_train, rows, input_cols)?;
```

### Example: Adding a New Activation Function

**Step 1: Implement it in activations.rs**

```rust
pub fn leaky_relu_activation(
    d_input: DevicePointer<f64>,
    d_output: DevicePointer<f64>,
    size: i32,
    module: &Module,
    stream: &Stream,
) -> CudaResult<()> {
    // Call your CUDA kernel (must be defined in gpu_kernels.cu)
    let func = module.get_function("leakyReluKernel")?;
    let block = (256, 1, 1);
    let grid_x = ((size as u32 + 255) / 256, 1, 1);

    unsafe {
        cust::launch!(func<<<grid_x, block, 0, stream>>>(d_input, d_output, size))?;
    }

    Ok(())
}

pub fn leaky_relu_derivative(
    d_z: DevicePointer<f64>,
    d_deriv: DevicePointer<f64>,
    size: i32,
    module: &Module,
    stream: &Stream,
) -> CudaResult<()> {
    let func = module.get_function("leakyReluDerivativeKernel")?;
    let block = (256, 1, 1);
    let grid_x = ((size as u32 + 255) / 256, 1, 1);

    unsafe {
        cust::launch!(func<<<grid_x, block, 0, stream>>>(d_z, d_deriv, size))?;
    }

    Ok(())
}
```

**Step 2: Use it in your network**

```rust
use crate::gpu_regression::activations::{leaky_relu_activation, leaky_relu_derivative};

let network = GpuNetworkBuilder::new()
    .add_linear("Input", 52, 64)?
    .add_activation("LeakyReLU", leaky_relu_activation, leaky_relu_derivative)
    .add_linear("Output", 64, 1)?
    .build();
```

## Key Data Structures

### LinearLayer

```rust
pub struct LinearLayer {
    pub name: String,
    pub input_size: usize,
    pub output_size: usize,
    pub weights: DeviceBuffer<f64>,  // On GPU
    pub biases: DeviceBuffer<f64>,   // On GPU
}

// Methods:
layer.weights();                  // Get weights pointer
layer.weights_mut();              // Get mutable weights
layer.save_parameters();          // Copy weights to host
layer.load_parameters(w, b);      // Load weights from host
```

### ActivationLayer

```rust
pub struct ActivationLayer {
    pub name: String,
    pub activation_fn: ActivationFn,      // Function pointer
    pub derivative_fn: ActivationDerivFn, // Derivative pointer
}
```

### GpuNetworkBuilder

```rust
let builder = GpuNetworkBuilder::new()
    .add_linear("L1", 10, 20)?
    .add_activation("Act1", fn_ref, deriv_ref)
    .add_linear("L2", 20, 1)?;

let network = builder.build();  // Returns GpuNetwork

// Access layers:
if let Some(NetworkLayer::Linear(l)) = network.get_layer(0) {
    println!("Layer 0 has {} inputs", l.input_size);
}
```

### TrainerConfig

```rust
let config = TrainerConfig {
    learning_rate: 0.0001,      // SGD learning rate
    epochs: 50000,              // Training iterations
    checkpoint_interval: 1000,  // Print & sync every N iters
    hidden_size: 64,            // Used for buffer allocation
};
```

## Training Loop Details

The trainer implements a standard backpropagation loop:

1. **Forward Pass**: Data flows through each layer:
   - Linear layer: Z = X @ W
   - ReLU layer: A = max(0, Z)

2. **Loss Computation**: MSE loss on final output

3. **Backward Pass**:
   - Compute gradients: dW = A^T @ error
   - Scale by learning rate: dW *= (lr / batch_size)
   - Update weights: W -= dW

4. **Synchronization**: Only at checkpoints (every N iterations) to minimize overhead

## Best Practices

### 1. Layer Naming
Give descriptive names for debugging:
```rust
.add_linear("Input_to_Hidden", 52, 64)?
.add_linear("Hidden_to_Output", 64, 1)?
```

### 2. Configuration Management
Centralize training config:
```rust
let config = TrainerConfig {
    learning_rate: 0.001,
    epochs: 20000,
    checkpoint_interval: 1000,
    hidden_size: 64,
};
```

### 3. Saving/Loading Weights
```rust
// Save
let weights = trainer.network.save_weights();

// Load (after building a new network with same architecture)
trainer.network.load_weights(weights)?;
```

### 4. Preprocessing
Normalize data on host before transfer:
```rust
let stats = compute_norm_stats(&x, rows, cols);
let x_norm = normalize_with_stats(&x, rows, cols, &stats);
let x_bias = add_bias_column(&x_norm, rows, cols);
```

## Comparison: Python vs Rust

| Feature | Python | Rust |
|---------|--------|------|
| Layer definition | `LinearLayer` class | `LinearLayer` struct |
| Network building | Manual layer.add() | `GpuNetworkBuilder` pattern |
| Training | `fit()` method | `GpuNetworkTrainer` class |
| Activation functions | Function pointers | Function pointers |
| Memory management | Automatic (CuPy) | Explicit (`DeviceBuffer`) |
| Error handling | Exceptions | `Result` type |
| GPU kernels | CuPy ops | Custom CUDA kernels |

## Extension Ideas

### Multi-Layer Support
Current trainer assumes 2-layer network. To support arbitrary depth:
- Store activations in cache during forward pass
- Iterate backward through layers

### Batch Normalization
- Add `BatchNormLayer` struct
- Implement running statistics on GPU
- Add scaling/centering in backward pass

### Different Optimizers
- Replace `scale_vector` + `update_weights` with SGD momentum, Adam, etc.
- Store optimizer state in a new `OptimizerConfig` struct

### Loss Function Abstraction
- Create `LossFunction` trait
- Implement MSE, CrossEntropy, HuberLoss
- Call `loss_prime()` in trainer backward pass

## Debugging

### Print Network Architecture
```rust
network.print_architecture();
// Output:
// === GPU Neural Network Architecture ===
// Layer 0: Input_to_Hidden
// Layer 1: Hidden_ReLU
// Layer 2: Hidden_to_Output
// =====================================
```

### Check Layer Dimensions
```rust
for (i, layer) in network.layers.iter().enumerate() {
    if let NetworkLayer::Linear(l) = layer {
        println!("Layer {}: {} -> {}", i, l.input_size, l.output_size);
    }
}
```

### Monitor Training Loss
The trainer prints epoch progress every checkpoint interval. Check stderr for GPU sync timing issues.

## Summary

This modular architecture makes it easy to:
- **Add layers**: Call `.add_linear()` or `.add_activation()`
- **Change activations**: Implement new function pair, use in builder
- **Experiment with depth**: Use preset builders or custom combinations
- **Save/load weights**: Single method calls
- **Configure training**: Adjust `TrainerConfig` fields

The design mirrors Python's flexibility while providing Rust's type safety and performance.
