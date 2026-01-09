use super::layers::*;
use crate::nn::LossFunction;
use crate::numeric::FloatingPoint;
use crate::tensor::math::TensorMath;
use crate::tensor::Tensor;
use std::f64::consts::PI;

use crate::nn::LayerData;
use crate::nn::ModelData;
use std::fs;
use std::fs::File;
use std::io;
use std::io::Write;
use std::path::Path;

use crate::nn::types::TrainingConfig;
use crate::nn::types::TrainingHook;

/// Feed-forward neural network container.
///
/// Holds an ordered list of `Layer` trait objects, the configured loss
/// function, and training metadata (parameter count, name/label, and
/// learning state). Instances are used to `predict`, `fit`, and
/// `save_model`.
pub struct NeuralNet<T, D>
where
    T: Tensor<D> + TensorMath<D, MathOutput = T> + 'static,
    D: FloatingPoint,
{
    pub layers: Vec<Box<dyn Layer<T, D>>>,
    pub loss_fn: Box<dyn LossFunction<T, D>>,
    pub parameter_count: u64,
    pub label: String,
    pub name: String,
    current_epoch: usize,
    current_lr: D,
}

impl<T, D> NeuralNet<T, D>
where
    T: Tensor<D> + TensorMath<D, MathOutput = T> + 'static,
    D: FloatingPoint,
{
    pub fn new(
        layers: Vec<Box<dyn Layer<T, D>>>,
        loss_fn: Box<dyn LossFunction<T, D>>,
        param_count: u64,
        label: String,
        name: String,
        current_epoch: usize,
        current_lr: D,
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
    pub fn add(&mut self, layer: Box<dyn Layer<T, D>>) {
        self.layers.push(layer);
    }

    /// Run a forward pass and return the network output for `input`.
    pub fn predict(&mut self, input: &T) -> Result<T, String> {
        let mut output = input.add(&T::zeroes(input.get_shape())).unwrap();

        for layer in &mut self.layers {
            output = layer.forward(&output, false).unwrap();
        }
        Ok(output)
    }

    pub fn fit<F>(
        &mut self,
        x_train: &T,
        y_train: &T,
        x_val: &T,
        y_val: &T,
        config: TrainingConfig<D>,
        mut hook_config: TrainingHook<F, Self, D>,
    ) -> Result<(), String>
    where
        F: FnMut(usize, D, D, D, &mut Self),
    {
        let lr_min = D::from_f64(1e-6);

        let total_timeline = D::from_u32(config.epochs as u32);
        let hook_interval = match config.epochs > hook_config.interval {
            true => hook_config.interval,
            false => config.epochs,
        };

        for i in config.epoch_offset..config.epochs {
            let global_i = D::from_u32(i as u32);

            self.current_epoch = i;

            let cos_term = ((D::from_f64(PI) * global_i) / total_timeline).cos();

            let decay_factor = D::from_f64(0.5) * (D::one() + cos_term);
            let current_lr = match config.lr_adjustment {
                true => lr_min + (config.base_lr - lr_min) * decay_factor,
                false => config.base_lr,
            };

            self.current_lr = current_lr;

            print!(
                "\rProcessing epoch: {}/{}, adjust lr set to {}, current lr: {current_lr}",
                self.current_epoch, config.epochs, config.lr_adjustment
            );
            io::stdout().flush().unwrap();

            let mut output = x_train.add(&T::zeroes(x_train.get_shape())).unwrap();
            for layer in &mut self.layers {
                output = layer.forward(&output, true).unwrap();
            }

            let mut v_output = x_val.add(&T::zeroes(x_val.get_shape())).unwrap();
            for layer in &mut self.layers {
                v_output = layer.forward(&v_output, false).unwrap();
            }
            T::synchronize();

            let err = self.loss_fn.loss(y_train, &output);
            let err_val = self.loss_fn.loss(y_val, &v_output);
            T::synchronize();

            let mut error_prime = self.loss_fn.loss_prime(y_train, &output).unwrap();

            for layer in self.layers.iter_mut().rev() {
                error_prime = layer.backward(&error_prime, current_lr).unwrap();
            }
            T::synchronize();

            // Hook (Periodic Reporting)
            if i == 0 || i % hook_interval == 0 {
                T::synchronize();
                let err = err.unwrap().sum().unwrap().get_data()[0];
                let err_val = err_val.unwrap().sum().unwrap().get_data()[0];

                (hook_config.callback)(i, err, err_val, current_lr, self);
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
                None => (Vec::<D>::new(), Vec::<u32>::new()),
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

        let path = Path::new(filepath);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap(); // Creates all directories if they don't exist
        }

        let mut file = File::create(filepath).unwrap();
        let _ = file.write_all(json_data.as_bytes());

        println!("Model successfully saved to {}", filepath);
    }
}
