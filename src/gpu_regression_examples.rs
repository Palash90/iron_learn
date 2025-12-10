// Example: Using the Modular GPU Network Architecture
//
// This file demonstrates how to build, train, and evaluate neural networks
// using the new modular structure that mirrors your Python implementation.

use crate::gpu_regression::builder::{build_simple_2layer, build_3layer, GpuNetworkBuilder};
use crate::gpu_regression::activations::{relu_activation, relu_derivative, sigmoid_activation, sigmoid_derivative};
use crate::gpu_regression::trainer::{GpuNetworkTrainer, TrainerConfig};
use crate::gpu_regression::{compute_norm_stats, normalize_with_stats, add_bias_column};
use crate::read_file::deserialize_data;
use crate::{Data, CpuTensor, GLOBAL_CONTEXT};
use cust::error::CudaResult;
use cust::module::Module;
use cust::stream::{Stream, StreamFlags};

/// Example 1: Simple 2-layer network for regression
pub fn example_simple_network() -> CudaResult<()> {
    println!("=== Example 1: Simple 2-Layer Network ===\n");

    // Get configuration from global context
    let learning_rate = GLOBAL_CONTEXT.get().unwrap().learning_rate;
    let epochs = GLOBAL_CONTEXT.get().unwrap().epochs;
    let data_path = &GLOBAL_CONTEXT.get().unwrap().data_path;

    // Load data
    let Data { cat_image: data, .. } = deserialize_data(data_path).unwrap();
    let rows = data.m as usize;
    let cols = data.n as usize;
    let input_cols = cols + 1; // Include bias

    // Preprocess: normalize and add bias column
    let stats = compute_norm_stats(&data.x, rows, cols);
    let x_norm = normalize_with_stats(&data.x, rows, cols, &stats);
    let x_bias = add_bias_column(&x_norm, rows, cols);

    // Load CUDA kernels
    let ptx = include_str!("../kernels/gpu_kernels.ptx");
    let module = Module::from_ptx(ptx, &[])?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Build a simple 2-layer network:
    // Input (52 features + bias) -> Hidden (64 units, ReLU) -> Output (1 unit)
    let network = build_simple_2layer(input_cols, 64, 1)?;
    network.print_architecture();

    // Configure training
    let config = TrainerConfig {
        learning_rate,
        epochs: epochs as usize,
        checkpoint_interval: 1000,
        hidden_size: 64,
    };

    // Create trainer and fit model
    let mut trainer = GpuNetworkTrainer::new(network, config, &module, stream);
    let (duration, final_loss) = trainer.fit(
        &x_bias,
        &data.y,
        rows,
        input_cols,
    )?;

    println!("\nTraining Results:");
    println!("  Time: {:?}", duration);
    println!("  Final Loss: {}", final_loss);

    // Evaluate on test set
    let test_rows = data.m_test as usize;
    if test_rows > 0 {
        let x_test_norm = normalize_with_stats(&data.x_test, test_rows, cols, &stats);
        let x_test_bias = add_bias_column(&x_test_norm, test_rows, cols);
        let test_mse = trainer.evaluate(&x_test_bias, &data.y_test, test_rows, input_cols)?;
        println!("  Test MSE: {}", test_mse);
    }

    Ok(())
}

/// Example 2: Deeper 3-layer network
pub fn example_3layer_network() -> CudaResult<()> {
    println!("\n=== Example 2: 3-Layer Network ===\n");

    let learning_rate = GLOBAL_CONTEXT.get().unwrap().learning_rate;
    let epochs = GLOBAL_CONTEXT.get().unwrap().epochs;
    let data_path = &GLOBAL_CONTEXT.get().unwrap().data_path;

    let Data { cat_image: data, .. } = deserialize_data(data_path).unwrap();
    let rows = data.m as usize;
    let cols = data.n as usize;
    let input_cols = cols + 1;

    let stats = compute_norm_stats(&data.x, rows, cols);
    let x_norm = normalize_with_stats(&data.x, rows, cols, &stats);
    let x_bias = add_bias_column(&x_norm, rows, cols);

    let ptx = include_str!("../kernels/gpu_kernels.ptx");
    let module = Module::from_ptx(ptx, &[])?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Build a 3-layer network:
    // Input (52 + bias) -> Hidden1 (128, ReLU) -> Hidden2 (64, ReLU) -> Output (1)
    let network = build_3layer(input_cols, 128, 64, 1)?;
    network.print_architecture();

    let config = TrainerConfig {
        learning_rate,
        epochs: epochs as usize,
        checkpoint_interval: 1000,
        hidden_size: 128,  // Use larger hidden size
    };

    let mut trainer = GpuNetworkTrainer::new(network, config, &module, stream);
    let (duration, final_loss) = trainer.fit(&x_bias, &data.y, rows, input_cols)?;

    println!("\nTraining Results:");
    println!("  Time: {:?}", duration);
    println!("  Final Loss: {}", final_loss);

    Ok(())
}

/// Example 3: Custom network with builder pattern
pub fn example_custom_network() -> CudaResult<()> {
    println!("\n=== Example 3: Custom Network with Builder ===\n");

    let learning_rate = GLOBAL_CONTEXT.get().unwrap().learning_rate;
    let epochs = GLOBAL_CONTEXT.get().unwrap().epochs;
    let data_path = &GLOBAL_CONTEXT.get().unwrap().data_path;

    let Data { cat_image: data, .. } = deserialize_data(data_path).unwrap();
    let rows = data.m as usize;
    let cols = data.n as usize;
    let input_cols = cols + 1;

    let stats = compute_norm_stats(&data.x, rows, cols);
    let x_norm = normalize_with_stats(&data.x, rows, cols, &stats);
    let x_bias = add_bias_column(&x_norm, rows, cols);

    let ptx = include_str!("../kernels/gpu_kernels.ptx");
    let module = Module::from_ptx(ptx, &[])?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Build a custom network with mixed activations:
    // Input -> ReLU -> Sigmoid -> ReLU -> Output
    let network = GpuNetworkBuilder::new()
        .add_linear("InputLayer", input_cols, 96)?
        .add_activation("ReLU1", relu_activation, relu_derivative)
        .add_linear("HiddenLayer1", 96, 48)?
        .add_activation("Sigmoid", sigmoid_activation, sigmoid_derivative)
        .add_linear("HiddenLayer2", 48, 24)?
        .add_activation("ReLU2", relu_activation, relu_derivative)
        .add_linear("OutputLayer", 24, 1)?
        .build();

    network.print_architecture();

    let config = TrainerConfig {
        learning_rate: learning_rate * 0.5,  // Use lower learning rate for this config
        epochs: epochs as usize,
        checkpoint_interval: 2000,
        hidden_size: 96,
    };

    let mut trainer = GpuNetworkTrainer::new(network, config, &module, stream);
    let (duration, final_loss) = trainer.fit(&x_bias, &data.y, rows, input_cols)?;

    println!("\nTraining Results:");
    println!("  Time: {:?}", duration);
    println!("  Final Loss: {}", final_loss);

    Ok(())
}

/// Example 4: Experiment with different architectures
pub fn example_architecture_comparison() -> CudaResult<()> {
    println!("\n=== Example 4: Architecture Comparison ===\n");

    let learning_rate = GLOBAL_CONTEXT.get().unwrap().learning_rate;
    let epochs = GLOBAL_CONTEXT.get().unwrap().epochs;
    let data_path = &GLOBAL_CONTEXT.get().unwrap().data_path;

    let Data { cat_image: data, .. } = deserialize_data(data_path).unwrap();
    let rows = data.m as usize;
    let cols = data.n as usize;
    let input_cols = cols + 1;

    let stats = compute_norm_stats(&data.x, rows, cols);
    let x_norm = normalize_with_stats(&data.x, rows, cols, &stats);
    let x_bias = add_bias_column(&x_norm, rows, cols);

    let ptx = include_str!("../kernels/gpu_kernels.ptx");
    let module = Module::from_ptx(ptx, &[])?;

    // Try multiple architectures with the same data
    let configs = vec![
        ("Small (32)", 32),
        ("Medium (64)", 64),
        ("Large (128)", 128),
    ];

    for (name, hidden_size) in configs {
        println!("\nTesting {}", name);

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        let network = build_simple_2layer(input_cols, hidden_size, 1)?;

        let config = TrainerConfig {
            learning_rate,
            epochs: epochs as usize,
            checkpoint_interval: 2000,
            hidden_size,
        };

        
        let mut trainer = GpuNetworkTrainer::new(network, config, &module, stream);
        match trainer.fit(&x_bias, &data.y, rows, input_cols) {
            Ok((duration, final_loss)) => {
                println!("  ✓ Time: {:?}, Final Loss: {}", duration, final_loss);
            }
            Err(e) => {
                println!("  ✗ Error: {:?}", e);
            }
        }
    }

    Ok(())
}

/// Example 5: How to add a new layer type (pseudo-code)
#[doc(hidden)]
pub fn example_adding_new_layer() {
    // Step 1: Define the layer in gpu_regression/layers.rs
    /*
    pub struct DropoutLayer {
        pub name: String,
        pub dropout_rate: f64,
    }

    impl DropoutLayer {
        pub fn new(name: &str, rate: f64) -> Self {
            DropoutLayer {
                name: name.to_string(),
                dropout_rate: rate,
            }
        }
    }
    */

    // Step 2: Add to NetworkLayer enum in gpu_regression/builder.rs
    /*
    pub enum NetworkLayer {
        Linear(LinearLayer),
        Activation(ActivationLayer),
        Dropout(DropoutLayer),
    }
    */

    // Step 3: Update GpuNetworkBuilder in gpu_regression/builder.rs
    /*
    impl GpuNetworkBuilder {
        pub fn add_dropout(mut self, name: &str, rate: f64) -> Self {
            let layer = DropoutLayer::new(name, rate);
            self.layers.push(NetworkLayer::Dropout(layer));
            self
        }
    }
    */

    // Step 4: Now you can use it immediately!
    /*
    let network = GpuNetworkBuilder::new()
        .add_linear("L1", 52, 64)?
        .add_activation("ReLU", relu_activation, relu_derivative)
        .add_dropout("Dropout1", 0.5)
        .add_linear("L2", 64, 1)?
        .build();
    */
}

/// Example 6: How to add a new activation function (pseudo-code)
#[doc(hidden)]
pub fn example_adding_activation() {
    // Step 1: Implement in gpu_regression/activations.rs
    /*
    pub fn gelu_activation(
        d_input: DevicePointer<f64>,
        d_output: DevicePointer<f64>,
        size: i32,
        module: &Module,
        stream: &Stream,
    ) -> CudaResult<()> {
        let func = module.get_function("geluKernel")?;
        let block = (256, 1, 1);
        let grid_x = ((size as u32 + 255) / 256, 1, 1);

        unsafe {
            cust::launch!(func<<<grid_x, block, 0, stream>>>(d_input, d_output, size))?;
        }
        Ok(())
    }

    pub fn gelu_derivative(
        d_z: DevicePointer<f64>,
        d_deriv: DevicePointer<f64>,
        size: i32,
        module: &Module,
        stream: &Stream,
    ) -> CudaResult<()> {
        // Similar implementation
        Ok(())
    }
    */

    // Step 2: Use it immediately in networks!
    /*
    let network = GpuNetworkBuilder::new()
        .add_linear("L1", 52, 64)?
        .add_activation("GELU", gelu_activation, gelu_derivative)
        .add_linear("L2", 64, 1)?
        .build();
    */
}
