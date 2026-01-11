use crate::NeuralNetBuilder;

use super::build_network::build_neural_net_from_config;

use crate::examples::contexts::GLOBAL_CONTEXT;
use crate::examples::read_file::deserialize_model;
use crate::nn::types::TrainingConfig;
use crate::nn::types::TrainingHook;
use crate::one_hot::one_hot_encode;
use crate::NeuralNet;

use rand::seq::SliceRandom;

pub fn run_for_grokking<T, D>() -> Result<(), String>
where
    T: crate::tensor::Tensor<D> + crate::tensor::math::TensorMath<D, MathOutput = T> + 'static,
    D: crate::numeric::FloatingPoint + 'static,
{
    let lr = GLOBAL_CONTEXT
        .get()
        .ok_or("GLOBAL_CONTEXT not initialized")?
        .learning_rate;
    let lr = D::from_f64(lr);
    let e = GLOBAL_CONTEXT.get().unwrap().epochs;
    let name = &GLOBAL_CONTEXT.get().unwrap().name;
    let distribution = &GLOBAL_CONTEXT.get().unwrap().distribution;
    let lr_adjustment = GLOBAL_CONTEXT.get().unwrap().lr_adjust;
    let restore = GLOBAL_CONTEXT.get().unwrap().restore;
    let predict_only = GLOBAL_CONTEXT.get().unwrap().predict_only;
    let weights_path = &GLOBAL_CONTEXT.get().unwrap().weights_path;

    let (l, epoch_offset, mut nn) = match !weights_path.is_empty() && restore {
        true => match deserialize_model::<D>(weights_path) {
            Some(model) => (
                model.saved_lr,
                model.epoch,
                NeuralNetBuilder::build_from_model(model),
            ),
            None => (lr, 0, build_neural_net_from_config(name, distribution)),
        },
        false => (lr, 0, build_neural_net_from_config(name, distribution)),
    };

    if !predict_only {
        // 1. Define the modular operation
        let mod_num = 7_u32;
        let p = mod_num; // The modulus

        let mut all_pairs = Vec::new();
        for a in 0..p {
            for b in 0..p {
                all_pairs.push((a, b));
            }
        }

        // 2. Shuffle and split for Grokking
        let mut rng = rand::rng();
        all_pairs.shuffle(&mut rng);

        // Grokking usually requires a small training fraction (e.g., 30% - 50%)
        let train_fraction = 0.5;
        let train_size = (all_pairs.len() as f64 * train_fraction) as usize;

        let train_pairs = &all_pairs[0..train_size];
        let val_pairs = &all_pairs[train_size..];

        // 3. Helper to build tensors for modular arithmetic
        let build_modular_xy = |pairs: &[(u32, u32)]| -> (Vec<u32>, Vec<u32>) {
            let mut inputs = Vec::new();
            let mut targets = Vec::new();
            for (a, b) in pairs {
                inputs.push(*a);
                inputs.push(*b);
                targets.push((*a + *b) % p); // Modular addition rule
            }
            (inputs, targets)
        };

        let (train_in, train_tar) = build_modular_xy(train_pairs);
        let (val_in, val_tar) = build_modular_xy(val_pairs);

        // multiplier is 2 here because we have pairs (a, b)
        let x_train = one_hot_encode(&train_in, mod_num, 2)?;
        let y_train = one_hot_encode::<T, D>(&train_tar, mod_num, 1)?;

        let x_val = one_hot_encode(&val_in, mod_num, 2)?;
        let y_val = one_hot_encode::<T, D>(&val_tar, mod_num, 1)?;

        // 4. Label Smoothing (Optional: often removed for pure grokking)
        let y_train = if false {
            // Set to true if you want to keep smoothing
            let epsilon = D::from_f64(0.1);
            let num_classes = D::from_u32(mod_num);
            let smooth_data: Vec<D> = y_train
                .get_data()
                .iter()
                .map(|&val| val * (D::one() - epsilon) + (epsilon / num_classes))
                .collect();
            T::new(y_train.get_shape().to_vec(), smooth_data)?
        } else {
            y_train
        };

        // 5. Training Setup
        let monitor = |epoch, loss: D, val_loss: D, _, nn: &mut NeuralNet<T, D>| {
            if loss.f64().is_nan() {
                panic!("Hit NaN");
            }
            println!("\tEpoch {epoch}: Train Loss = {loss:.6}, Val Loss = {val_loss:.6}");

            // Save model periodically
            if epoch % 1000 == 0 {
                nn.save_model(weights_path);
            }
        };

        let config = TrainingConfig {
            epochs: e as usize,
            epoch_offset,
            base_lr: l,
            lr_adjustment,
            weight_normalization: true,
        };

        // Increase interval for cleaner logs during long grokking runs
        let hook_config = TrainingHook::new(500, monitor);

        println!(
            "Starting Grokking Experiment: {} training samples",
            train_size
        );
        nn.fit(&x_train, &y_train, &x_val, &y_val, config, hook_config)?;
    }

    Ok(())
}
