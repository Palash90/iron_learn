use super::builder::{GpuNetwork, NetworkLayer};
use super::functions::*;
use super::layers::LinearLayer;
use crate::{CpuTensor, Data};
use cust::error::CudaResult;
use cust::memory::DeviceBuffer;
use cust::module::Module;
use cust::prelude::*;
use cust::stream::Stream;
use std::time::Instant;

/// Configuration for training
pub struct TrainerConfig {
    pub learning_rate: f64,
    pub epochs: usize,
    pub checkpoint_interval: usize, // Sync and print every N iterations
    pub hidden_size: usize,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        TrainerConfig {
            learning_rate: 0.001,
            epochs: 10000,
            checkpoint_interval: 1000,
            hidden_size: 64,
        }
    }
}

/// A GPU neural network trainer that handles the training loop
pub struct GpuNetworkTrainer<'a> {
    pub network: GpuNetwork,
    pub config: TrainerConfig,
    pub module: &'a cust::module::Module,
    pub stream: cust::stream::Stream,
}

impl<'a> GpuNetworkTrainer<'a> {
    /// Create a new trainer with a network and configuration
    pub fn new(
        network: GpuNetwork,
        config: TrainerConfig,
        module: &'a Module,
        stream: Stream,
    ) -> Self {
        GpuNetworkTrainer {
            network,
            config,
            module,
            stream,
        }
    }

    /// Train the network on the provided data (simple 2-layer version)
    /// Returns (training_time, final_loss)
    pub fn fit(
        &mut self,
        x_train: &[f64],
        y_train: &[f64],
        rows: usize,
        input_cols: usize, // includes bias
    ) -> CudaResult<(std::time::Duration, f64)> {
        let start = Instant::now();
        let module = &self.module;
        let stream = &self.stream;
        let lr = self.config.learning_rate;
        let epochs = self.config.epochs;
        let checkpoint_interval = self.config.checkpoint_interval;
        let hidden_size = self.config.hidden_size;

        // Transfer training data to GPU (once)
        let d_x = DeviceBuffer::from_slice(x_train)?;
        let d_y = DeviceBuffer::from_slice(y_train)?;

        // Get mutable references to weight layers
        // For a 2-layer network: Layer 0 = Linear (W1), Layer 1 = ReLU, Layer 2 = Linear (W2)
        let d_w1 = if let Some(NetworkLayer::Linear(ref layer)) = self.network.get_layer(0) {
            layer.weights().as_device_ptr()
        } else {
            panic!("Layer 0 is not linear - expected 2-layer network structure");
        };

        let d_w2 = if let Some(NetworkLayer::Linear(ref layer)) = self.network.get_layer(2) {
            layer.weights().as_device_ptr()
        } else {
            panic!("Layer 2 is not linear - expected 2-layer network structure");
        };

        // Pre-allocate GPU buffers for forward/backward passes
        let d_z1 = DeviceBuffer::<f64>::zeroed(rows * hidden_size)?;
        let d_a1 = DeviceBuffer::<f64>::zeroed(rows * hidden_size)?;
        let d_z2 = DeviceBuffer::<f64>::zeroed(rows)?;
        let d_loss = DeviceBuffer::<f64>::zeroed(rows)?;
        let d_dw2 = DeviceBuffer::<f64>::zeroed(hidden_size)?;
        let d_delta = DeviceBuffer::<f64>::zeroed(rows * hidden_size)?;
        let d_relu_deriv = DeviceBuffer::<f64>::zeroed(rows * hidden_size)?;
        let d_dw1 = DeviceBuffer::<f64>::zeroed(input_cols * hidden_size)?;
        let d_w2_t = DeviceBuffer::<f64>::zeroed(hidden_size)?;
        let d_x_t = DeviceBuffer::<f64>::zeroed(input_cols * rows)?;

        for iteration in 0..epochs {
            // === Forward Pass ===
            // Z1 = X @ W1
            matrix_mul(
                module,
                stream,
                d_x.as_device_ptr(),
                d_w1,
                d_z1.as_device_ptr(),
                rows as i32,
                hidden_size as i32,
                input_cols as i32,
            )?;

            // A1 = relu(Z1)
            relu_kernel(
                module,
                stream,
                d_z1.as_device_ptr(),
                d_a1.as_device_ptr(),
                (rows * hidden_size) as i32,
            )?;

            // Z2 = A1 @ W2
            matrix_mul(
                module,
                stream,
                d_a1.as_device_ptr(),
                d_w2,
                d_z2.as_device_ptr(),
                rows as i32,
                1,
                hidden_size as i32,
            )?;

            // === Compute Loss ===
            vector_sub(
                module,
                stream,
                d_z2.as_device_ptr(),
                d_y.as_device_ptr(),
                d_loss.as_device_ptr(),
                rows as i32,
                1,
            )?;

            // === Backward Pass ===
            // dW2 = A1^T @ loss
            grad_gemv_xt(
                module,
                stream,
                d_a1.as_device_ptr(),
                d_loss.as_device_ptr(),
                d_dw2.as_device_ptr(),
                rows as i32,
                hidden_size as i32,
            )?;

            // Scale dW2 by learning rate
            scale_vector(
                module,
                stream,
                d_dw2.as_device_ptr(),
                lr / rows as f64,
                hidden_size as i32,
            )?;

            // Update W2
            update_weights(
                module,
                stream,
                d_w2,
                d_dw2.as_device_ptr(),
                hidden_size as i32,
            )?;

            // === Backprop to Hidden ===
            transpose_naive(
                module,
                stream,
                d_w2,
                d_w2_t.as_device_ptr(),
                hidden_size as i32,
                1,
            )?;

            matrix_mul(
                module,
                stream,
                d_loss.as_device_ptr(),
                d_w2_t.as_device_ptr(),
                d_delta.as_device_ptr(),
                rows as i32,
                hidden_size as i32,
                1,
            )?;

            // Apply ReLU derivative
            relu_derivative(
                module,
                stream,
                d_z1.as_device_ptr(),
                d_relu_deriv.as_device_ptr(),
                (rows * hidden_size) as i32,
            )?;

            hadamard_prod(
                module,
                stream,
                d_delta.as_device_ptr(),
                d_relu_deriv.as_device_ptr(),
                d_delta.as_device_ptr(),
                (rows * hidden_size) as i32,
            )?;

            // dW1 = X^T @ delta
            transpose_naive(
                module,
                stream,
                d_x.as_device_ptr(),
                d_x_t.as_device_ptr(),
                rows as i32,
                input_cols as i32,
            )?;

            matrix_mul(
                module,
                stream,
                d_x_t.as_device_ptr(),
                d_delta.as_device_ptr(),
                d_dw1.as_device_ptr(),
                input_cols as i32,
                hidden_size as i32,
                rows as i32,
            )?;

            // Scale dW1 and update W1
            scale_vector(
                module,
                stream,
                d_dw1.as_device_ptr(),
                lr / rows as f64,
                (input_cols * hidden_size) as i32,
            )?;

            update_weights(
                module,
                stream,
                d_w1,
                d_dw1.as_device_ptr(),
                (input_cols * hidden_size) as i32,
            )?;

            // Checkpoint synchronization
            if iteration % checkpoint_interval == 0 {
                stream.synchronize()?;
                println!("Epoch {} / {} complete", iteration, epochs);
            }
        }

        // Final synchronization
        stream.synchronize()?;
        let duration = start.elapsed();

        // Compute final loss
        let mut loss_host = vec![0.0f64; rows];
        d_loss.copy_to(&mut loss_host)?;
        let final_loss: f64 = loss_host.iter().sum::<f64>() / (rows as f64);

        Ok((duration, final_loss))
    }

    /// Evaluate network on test data
    pub fn evaluate(
        &self,
        x_test: &[f64],
        y_test: &[f64],
        rows: usize,
        input_cols: usize,
    ) -> CudaResult<f64> {
        let module = &self.module;
        let stream = &self.stream;
        let hidden_size = self.config.hidden_size;

        let d_w1 = if let Some(NetworkLayer::Linear(ref layer)) = self.network.get_layer(0) {
            layer.weights().as_device_ptr()
        } else {
            panic!("Layer 0 is not linear - expected 2-layer network structure");
        };

        let d_w2 = if let Some(NetworkLayer::Linear(ref layer)) = self.network.get_layer(2) {
            layer.weights().as_device_ptr()
        } else {
            panic!("Layer 2 is not linear - expected 2-layer network structure");
        };

        // Transfer test data to GPU
        let d_x_test = DeviceBuffer::from_slice(x_test)?;
        let d_y_test = DeviceBuffer::from_slice(y_test)?;

        // Forward pass on test data
        let d_z1 = DeviceBuffer::<f64>::zeroed(rows * hidden_size)?;
        let d_a1 = DeviceBuffer::<f64>::zeroed(rows * hidden_size)?;
        let d_z2 = DeviceBuffer::<f64>::zeroed(rows)?;

        matrix_mul(
            module,
            stream,
            d_x_test.as_device_ptr(),
            d_w1,
            d_z1.as_device_ptr(),
            rows as i32,
            hidden_size as i32,
            input_cols as i32,
        )?;

        relu_kernel(
            module,
            stream,
            d_z1.as_device_ptr(),
            d_a1.as_device_ptr(),
            (rows * hidden_size) as i32,
        )?;

        matrix_mul(
            module,
            stream,
            d_a1.as_device_ptr(),
            d_w2,
            d_z2.as_device_ptr(),
            rows as i32,
            1,
            hidden_size as i32,
        )?;

        // Compute loss on test set
        let d_loss = DeviceBuffer::<f64>::zeroed(rows)?;
        vector_sub(
            module,
            stream,
            d_z2.as_device_ptr(),
            d_y_test.as_device_ptr(),
            d_loss.as_device_ptr(),
            rows as i32,
            1,
        )?;

        stream.synchronize()?;
        let mut loss_host = vec![0.0f64; rows];
        d_loss.copy_to(&mut loss_host)?;

        let mut d_z2_host = vec![0.0f64; rows];
        d_z2.copy_to(&mut d_z2_host)?;
        println!("{:?}", d_z2_host);
        println!("{:?}", y_test);
        let mse = loss_host.iter().map(|l| l * l).sum::<f64>() / (rows as f64);

        Ok(mse)
    }
}
