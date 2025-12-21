use super::layers::*;
use super::NeuralNetDataType;
use crate::neural_network::LossFunction;
use crate::numeric::Numeric;
use crate::tensor::math::TensorMath;
use crate::tensor::Tensor;
use std::f32::consts::PI;

pub struct NeuralNet<T>
where
    T: Tensor<NeuralNetDataType> + TensorMath<NeuralNetDataType, MathOutput = T> + 'static,
{
    pub layers: Vec<Box<dyn Layer<T>>>,
    pub loss_fn: Box<dyn LossFunction<NeuralNetDataType, T>>,
}

impl<T> NeuralNet<T>
where
    T: Tensor<NeuralNetDataType> + TensorMath<NeuralNetDataType, MathOutput = T> + 'static,
{
    pub fn add(&mut self, layer: Box<dyn Layer<T>>) {
        self.layers.push(layer);
    }

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
    ) -> Result<(), String>
    where
        F: FnMut(usize, NeuralNetDataType, &mut Self),
    {
        let lr_min = 1e-6;

        for i in 0..epochs {
            println!("Epoch {}", i);

            let cos_term = ((PI as NeuralNetDataType * i as NeuralNetDataType)
                / ((epochs + epoch_offset) as NeuralNetDataType))
                .cos();
            println!("cos {}", cos_term);

            let decay_factor = 0.5 * (1.0 + cos_term);
            let current_lr = match lr_adjustment {
                true => lr_min + (base_lr.f32() - lr_min) * decay_factor,
                false => base_lr,
            };

            println!("Current lr {}", current_lr);

            let mut output = x_train.add(&T::zeroes(x_train.get_shape())).unwrap();
            for layer in &mut self.layers {
                output = layer.forward(&output).unwrap();
            }
            println!("Layer forwarded");

            let err = self.loss_fn.loss(y_train, &output);

            println!("Error calculated");

            let mut error_prime = self.loss_fn.loss_prime(y_train, &output).unwrap();
            println!("Error Prime Calculated");

            for layer in self.layers.iter_mut().rev() {
                println!("\t\tLayer {} backward started", layer.name());
                error_prime = layer.backward(&error_prime, current_lr).unwrap();
                println!("\t\tLayer {} backward completed", layer.name());
            }
            println!("Backward Error Primed");

            let hook_interval = match epochs > 1 {
                true => 1,
                false => epochs - 1,
            };

            // Hook (Periodic Reporting)
            if i == 0 || i % hook_interval == 0 {
                x_train.synchronize();
                let e_1 = err.unwrap();
                println!("E_1 calc");

                let e_2 = e_1.sum().unwrap();
                println!("E_2 calc");

                let err_val = e_2.get_data()[0];
                hook(i, err_val, self);
            }

            println!();
        }

        x_train.synchronize();
        Ok(())
    }

    pub fn save_weights(&self, filepath: &str) {
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(w) = layer.get_parameters() {
                println!(
                    "Layer {} ({}) Weights saved. (Shape: {:?})",
                    i,
                    layer.name(),
                    w.len()
                );
            }
        }

        println!("To be saved in {}", filepath)
    }
}
