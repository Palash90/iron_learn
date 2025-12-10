# Quick Reference - Modular GPU Networks

## Import What You Need
```rust
use crate::gpu_regression::builder::{GpuNetworkBuilder, build_simple_2layer};
use crate::gpu_regression::activations::{relu_activation, relu_derivative};
use crate::gpu_regression::trainer::{GpuNetworkTrainer, TrainerConfig};
use crate::gpu_regression::{compute_norm_stats, normalize_with_stats, add_bias_column};
```

## 3-Step Neural Network Training

### Step 1: Build Network
```rust
// Option A: Use preset
let network = build_simple_2layer(input_size, hidden_size, output_size)?;

// Option B: Custom builder
let network = GpuNetworkBuilder::new()
    .add_linear("Layer1", input_size, hidden_size)?
    .add_activation("ReLU", relu_activation, relu_derivative)
    .add_linear("Layer2", hidden_size, output_size)?
    .build();
```

### Step 2: Configure Training
```rust
let config = TrainerConfig {
    learning_rate: 0.001,
    epochs: 10000,
    checkpoint_interval: 1000,
    hidden_size: 64,
};
```

### Step 3: Train
```rust
let mut trainer = GpuNetworkTrainer::new(network, config, module, stream);
trainer.fit(&x_train, &y_train, num_samples, input_dim)?;
```

## Common Tasks

### Print Network Architecture
```rust
network.print_architecture();
```

### Evaluate on Test Set
```rust
let test_mse = trainer.evaluate(&x_test, &y_test, num_test, input_dim)?;
```

### Save/Load Weights
```rust
// Save
let weights = trainer.network.save_weights();

// Load
trainer.network.load_weights(weights)?;
```

### Try Different Architectures
```rust
// Small network
let net1 = build_simple_2layer(52, 32, 1)?;

// Medium network  
let net2 = build_simple_2layer(52, 64, 1)?;

// Deep network
let net3 = build_3layer(52, 128, 64, 1)?;

// Custom network
let net4 = GpuNetworkBuilder::new()
    .add_linear("L1", 52, 256)?
    .add_activation("ReLU1", relu_activation, relu_derivative)
    .add_linear("L2", 256, 128)?
    .add_activation("ReLU2", relu_activation, relu_derivative)
    .add_linear("L3", 128, 1)?
    .build();
```

## Module Locations

| Module | File | Purpose |
|--------|------|---------|
| `layers` | `gpu_regression/layers.rs` | LinearLayer, ActivationLayer |
| `activations` | `gpu_regression/activations.rs` | relu, sigmoid, tanh, etc. |
| `builder` | `gpu_regression/builder.rs` | GpuNetworkBuilder, presets |
| `trainer` | `gpu_regression/trainer.rs` | GpuNetworkTrainer, config |
| `gpu_regression_helpers` | `gpu_regression/gpu_regression_helpers.rs` | Kernel wrappers |

## Add New Activation Function

1. **Define in `activations.rs`:**
```rust
pub fn my_activation(d_input, d_output, size, module, stream) -> CudaResult<()> {
    // Your kernel call here
}

pub fn my_derivative(d_z, d_deriv, size, module, stream) -> CudaResult<()> {
    // Derivative kernel
}
```

2. **Use in network:**
```rust
.add_activation("MyAct", my_activation, my_derivative)
```

## Add New Layer Type

1. **Define in `layers.rs`:**
```rust
pub struct MyLayer {
    pub name: String,
    // Your fields...
}
```

2. **Add to builder enum in `builder.rs`:**
```rust
pub enum NetworkLayer {
    Linear(LinearLayer),
    Activation(ActivationLayer),
    MyLayer(MyLayer),  // <- Add this
}
```

3. **Add builder method in `GpuNetworkBuilder`:**
```rust
pub fn add_my_layer(mut self, name: &str, ...) -> Self {
    let layer = MyLayer::new(...);
    self.layers.push(NetworkLayer::MyLayer(layer));
    self
}
```

4. **Use it:**
```rust
.add_my_layer("MyName", params)
```

## Data Preprocessing (CPU-side)
```rust
// Compute statistics
let stats = compute_norm_stats(&x, rows, cols);

// Normalize
let x_norm = normalize_with_stats(&x, rows, cols, &stats);

// Add bias column
let x_bias = add_bias_column(&x_norm, rows, cols);

// Now transfer to GPU trainer
trainer.fit(&x_bias, &y, rows, x_bias.cols)?;
```

## Preset Network Builders
```rust
// 2-layer: Input -> Hidden (ReLU) -> Output
build_simple_2layer(input_size, hidden_size, output_size)?

// 3-layer: Input -> H1 (ReLU) -> H2 (ReLU) -> Output
build_3layer(input_size, h1_size, h2_size, output_size)?

// 5-layer deep: Input -> H1 (ReLU) -> H2 (ReLU) -> H3 (ReLU) -> H4 (ReLU) -> H5 (ReLU) -> Output
build_deep_network(input_size, hidden_size, output_size)?
```

## TrainerConfig Fields
```rust
pub struct TrainerConfig {
    pub learning_rate: f64,          // SGD learning rate (0.001)
    pub epochs: usize,                // Iterations to train (10000)
    pub checkpoint_interval: usize,   // Sync every N iters (1000)
    pub hidden_size: usize,           // For buffer allocation (64)
}
```

## Error Handling
```rust
// Most functions return CudaResult<()> or CudaResult<T>
match trainer.fit(&x, &y, rows, cols) {
    Ok((duration, loss)) => println!("Loss: {}", loss),
    Err(e) => eprintln!("Error: {:?}", e),
}
```

## Common Patterns

### Hyperparameter Search
```rust
for lr in vec![0.0001, 0.001, 0.01] {
    for hidden in vec![32, 64, 128] {
        let network = build_simple_2layer(52, hidden, 1)?;
        let mut config = TrainerConfig::default();
        config.learning_rate = lr;
        config.hidden_size = hidden;
        
        let mut trainer = GpuNetworkTrainer::new(network, config, &module, &stream);
        let (time, loss) = trainer.fit(&x, &y, rows, cols)?;
        println!("lr={}, hidden={}, loss={}", lr, hidden, loss);
    }
}
```

### Comparing Architectures
```rust
let configs = vec![
    ("Small", build_simple_2layer(52, 32, 1)?),
    ("Medium", build_simple_2layer(52, 64, 1)?),
    ("Large", build_simple_2layer(52, 128, 1)?),
    ("Deep", build_3layer(52, 128, 64, 1)?),
];

for (name, network) in configs {
    let mut trainer = GpuNetworkTrainer::new(network, config.clone(), &module, &stream);
    let (_, loss) = trainer.fit(&x, &y, rows, cols)?;
    println!("{}: loss={}", name, loss);
}
```

## File Structure at a Glance
```
src/gpu_regression/
├── layers.rs          <- Layer definitions
├── activations.rs     <- Activation functions  
├── builder.rs         <- Network building
├── trainer.rs         <- Training loop
├── gpu_regression_helpers.rs <- Kernel wrappers
└── mod.rs            <- Exports
```

## Remember
- **Layers are just structs** - Easy to add new types
- **Activations are function pointers** - Easy to swap in/out
- **Builder is fluent** - Chain `.add_linear()` and `.add_activation()`
- **Trainer is simple** - One `fit()` call, one `evaluate()` call
- **Config is explicit** - All parameters in `TrainerConfig`

**Your network is now as flexible as Python! 🚀**
