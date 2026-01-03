use super::layers::*;
use super::NeuralNetDataType;
use crate::nn::LossFunction;
use crate::numeric::Numeric;
use crate::tensor::math::TensorMath;
use crate::tensor::Tensor;
use std::f32::consts::PI;

use crate::nn::LayerData;
use crate::nn::ModelData;
use std::fs::File;
use std::io;
use std::io::Write;

/// Feed-forward neural network container.
///
/// Holds an ordered list of `Layer` trait objects, the configured loss
/// function, and training metadata (parameter count, name/label, and
/// learning state). Instances are used to `predict`, `fit`, and
/// `save_model`.
pub struct NeuralNet<T>
where
    T: Tensor<NeuralNetDataType> + TensorMath<NeuralNetDataType, MathOutput = T> + 'static,
{
    pub layers: Vec<Box<dyn Layer<T>>>,
    pub loss_fn: Box<dyn LossFunction<NeuralNetDataType, T>>,
    pub parameter_count: u64,
    pub label: String,
    pub name: String,
    current_epoch: usize,
    current_lr: NeuralNetDataType,
}

impl<T> NeuralNet<T>
where
    T: Tensor<NeuralNetDataType> + TensorMath<NeuralNetDataType, MathOutput = T> + 'static,
{
    pub fn new(
        layers: Vec<Box<dyn Layer<T>>>,
        loss_fn: Box<dyn LossFunction<NeuralNetDataType, T>>,
        param_count: u64,
        label: String,
        name: String,
        current_epoch: usize,
        current_lr: NeuralNetDataType,
    ) -> Self {
        Self {
            layers,
            loss_fn,
            parameter_count: param_count,
            label,
            name,
            current_epoch,
            current_lr,
        }
    }
    /// Append a layer to the network.
    ///
    /// The provided box must implement the `Layer` trait for the network's
    /// tensor type `T`.
    pub fn add(&mut self, layer: Box<dyn Layer<T>>) {
        self.layers.push(layer);
    }

    /// Run a forward pass and return the network output for `input`.
    pub fn predict(&mut self, input: &T) -> Result<T, String> {
        let mut output = input.add(&T::zeroes(input.get_shape())).unwrap();

        for layer in &mut self.layers {
            output = layer.forward(&output).unwrap();
        }
        Ok(output)
    }

    pub fn fit<F>(
        &mut self,
        x_train: &T,
        y_train: &T,
        epochs: usize,
        epoch_offset: usize,
        base_lr: NeuralNetDataType,
        lr_adjustment: bool,
        mut hook: F,
        hook_interval: usize,
    ) -> Result<(), String>
    where
        F: FnMut(usize, NeuralNetDataType, NeuralNetDataType, &mut Self),
    {
        let lr_min = 1e-6;

        let total_timeline = epochs as NeuralNetDataType;
        let hook_interval = match epochs > hook_interval {
            true => hook_interval,
            false => epochs,
        };

        for i in epoch_offset..epochs {
            let global_i = i as NeuralNetDataType;

            self.current_epoch = i;

            let cos_term = ((PI as NeuralNetDataType * global_i) / total_timeline).cos();

            let decay_factor = 0.5 * (1.0 + cos_term);
            let current_lr = match lr_adjustment {
                true => lr_min + (base_lr.f32() - lr_min) * decay_factor,
                false => base_lr,
            };

            self.current_lr = current_lr;

            print!(
                "\rProcessing epoch: {}/{epochs}, adjust lr set to {lr_adjustment}, current lr: {current_lr}", self.current_epoch
            );
            io::stdout().flush().unwrap();

            let mut output = x_train.add(&T::zeroes(x_train.get_shape())).unwrap();
            for layer in &mut self.layers {
                output = layer.forward(&output).unwrap();
            }
            T::synchronize();

            let err = self.loss_fn.loss(y_train, &output);
            T::synchronize();

            let mut error_prime = self.loss_fn.loss_prime(y_train, &output).unwrap();

            for layer in self.layers.iter_mut().rev() {
                error_prime = layer.backward(&error_prime, current_lr).unwrap();
            }
            T::synchronize();

            // Hook (Periodic Reporting)
            if i == 0 || i % hook_interval == 0 {
                T::synchronize();
                let err_val = err.unwrap().sum().unwrap().get_data()[0];

                hook(i, err_val, current_lr, self);
            }
        }

        T::synchronize();
        Ok(())
    }
    /// Serialize and write the model weights and metadata to `filepath`.
    pub fn save_model(&self, filepath: &str) {
        let mut model_storage = ModelData {
            name: self.name.clone(),
            parameter_count: self.parameter_count,
            layers: Vec::new(),
            epoch: self.current_epoch,
            saved_lr: self.current_lr,
        };

        for (i, layer) in self.layers.iter().enumerate() {
            let (w, s) = match layer.get_parameters() {
                Some(wt) => (wt.get_data().to_vec(), wt.get_shape().to_vec()),
                None => (Vec::<NeuralNetDataType>::new(), Vec::<u32>::new()),
            };

            let layer_info = LayerData {
                name: layer.name().to_string(),
                index: i,
                weights: w,
                shape: s,
                layer_type: layer.layer_type().clone(),
            };

            model_storage.layers.push(layer_info);
        }

        let json_data = serde_json::to_string_pretty(&model_storage)
            .expect("Failed to serialize model weights");

        let mut file = File::create(filepath).unwrap();
        let _ = file.write_all(json_data.as_bytes());

        println!("Model successfully saved to {}", filepath);
    }
}
