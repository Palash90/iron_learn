use crate::nn::loss_functions::BinaryCrossEntropy;
use crate::nn::{DistributionType, LayerType};
use crate::NeuralNetBuilder;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufRead, BufReader};

use std::thread;
use std::time::Duration;

use crate::numeric::FloatingPoint;
use crate::tensor::math::TensorMath;
use crate::Tensor;

use crate::examples::contexts::GLOBAL_CONTEXT;
use crate::examples::read_file::deserialize_model;

pub fn run_trigram_generator<T, D>() -> Result<(), String>
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
    let sleep_time = GLOBAL_CONTEXT.get().unwrap().sleep_time;
    let hidden_length = GLOBAL_CONTEXT.get().unwrap().hidden_layer_length;
    let name = &GLOBAL_CONTEXT.get().unwrap().name;
    let distribution = &GLOBAL_CONTEXT.get().unwrap().distribution;
    let weights_path = &GLOBAL_CONTEXT.get().unwrap().weights_path;
    let weights_path = name.to_owned() + "/" + &weights_path;
    let lr_adjustment = GLOBAL_CONTEXT.get().unwrap().lr_adjust;
    let restore = GLOBAL_CONTEXT.get().unwrap().restore;

    let file = File::open("data/names.txt").expect("File could not be opened");
    let reader = BufReader::new(file);

    // Collect lines into a Vec<String>, handling possible I/O errors
    let lines: Result<Vec<String>, io::Error> = reader
        .lines() // Iterator over Result<String, io::Error>
        .collect();

    let lines = match lines {
        Ok(lines) => lines,
        Err(e) => {
            eprintln!("Error reading file: {}", e);
            return Err("Error reading file".to_string());
        }
    };

    let names = lines;

    // 1. Build Vocabulary
    let mut chars: Vec<char> = names
        .iter()
        .flat_map(|s| s.chars())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    chars.sort();
    chars.insert(0, '.'); // Start/End token

    println!("Vocabulary Size: {}, {:?}", chars.len(), chars);

    let stoi: HashMap<char, usize> = chars.iter().enumerate().map(|(i, &c)| (c, i)).collect();
    let itos: HashMap<usize, char> = chars.iter().enumerate().map(|(i, &c)| (i, c)).collect();
    let vocab_size = chars.len() as u32;

    // 2. Create Training Data (Bigrams)
    let mut inputs = Vec::new();
    let mut targets = Vec::new();

    for name in names {
        let full_name = format!("..{}..", name);
        let chars_vec: Vec<char> = full_name.chars().collect();
        for window in chars_vec.windows(3) {
            let ix1 = stoi[&window[0]];
            let ix2 = stoi[&window[1]];

            // Create One-Hot for Input
            let mut oh = vec![D::zero(); vocab_size as usize];
            oh[ix1] = D::one();
            oh[ix2 + vocab_size as usize] = D::one();
            inputs.extend(oh);

            // Target (Simplified: one-hot for MSE)
            let mut target_oh = vec![D::zero(); vocab_size as usize];
            target_oh[ix2] = D::one();
            targets.extend(target_oh);
        }
    }

    let num_samples = (inputs.len() as u32) / (vocab_size * 2);
    let x_train = T::new(vec![num_samples, vocab_size * 2], inputs)?;
    let y_train = T::new(vec![num_samples, vocab_size], targets)?;

    let loss_function_instance = Box::new(BinaryCrossEntropy);

    let (l, epoch_offset, mut nn) = match !weights_path.is_empty() && restore {
        true => match deserialize_model::<D>(&weights_path) {
            Some(model) => (
                model.saved_lr.clone(),
                model.epoch.clone(),
                NeuralNetBuilder::build_from_model(model, loss_function_instance),
            ),
            None => (
                lr,
                0,
                define_neural_net::<T, D>(hidden_length, vocab_size, distribution)
                    .build(loss_function_instance, name),
            ),
        },
        false => (
            lr,
            0,
            define_neural_net::<T, D>(hidden_length, vocab_size, distribution)
                .build(loss_function_instance, name),
        ),
    };

    // 4. Training (Small number of epochs for demo)
    nn.fit(
        &x_train,
        &y_train,
        e as usize,
        epoch_offset,
        l,
        lr_adjustment,
        |epoch, loss, _, nn| {
            println!("\tEpoch {epoch}: Loss (BCE) = {loss:.8}");
            nn.save_model(&weights_path);

            if sleep_time > 0 && epoch != 0 {
                println!("Taking a nap");
                thread::sleep(Duration::from_secs(5));
                println!("Awake again");
            }
        },
        1000,
    )?;

    println!("\nGenerated Names:");
    for _ in 0..10 {
        let mut context = ('.', '.');
        let mut name = String::new();

        loop {
            let mut input_vec = vec![D::zero(); (vocab_size * 2) as usize];
            input_vec[stoi[&context.0]] = D::one();
            input_vec[stoi[&context.1] + vocab_size as usize] = D::one();

            let input_tensor = T::new(vec![1, vocab_size * 2], input_vec)?;
            let preds = nn.predict(&input_tensor)?;
            let data = preds.get_data();

            // --- Weighted Random Sampling ---
            // 1. Convert logits/scores to positive weights (exponentiation)
            let temparature = 0.4;
            let mut weights: Vec<f64> = data
                .iter()
                .map(|val| (val.f64() / temparature).exp())
                .collect();

            if name.len() > 4 {
                weights[0] *= 2.0; // Double the chance of ending
            }
            if name.len() > 6 {
                weights[0] *= 10.0; // Force an end
            }

            let total_weight: f64 = weights.iter().sum();

            // 2. Generate a pseudo-random point
            let rng_seed = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as f64;
            let random_point = (rng_seed % 1000.0 / 1000.0) * total_weight;

            let mut next_ix = 0;
            let mut cumulative_weight = 0.0;
            let mut next_ix = 0;
            for (i, &w) in weights.iter().enumerate() {
                cumulative_weight += w;
                if cumulative_weight >= random_point {
                    next_ix = i;
                    break;
                }
            }

            let next_char = itos[&next_ix];
            if next_char == '.' {
                if name.len() < 3 {
                    continue;
                } else {
                    break;
                }
            }

            name.push(next_char);

            context = (context.1, next_char);

            if name.len() > 20 {
                break;
            }
        }
        println!("> {}", name);
    }

    Ok(())
}

fn define_neural_net<T, D>(
    hl: u32,
    input: u32,
    distribution: &DistributionType,
) -> NeuralNetBuilder<T, D>
where
    T: Tensor<D> + TensorMath<D, MathOutput = T> + 'static,
    D: FloatingPoint + 'static,
{
    let mut nn = NeuralNetBuilder::<T, D>::new();

    let input_size = input * 2;

    let layers = [
        (input_size, hl, LayerType::Tanh, "Input", "AL 1"),
        (hl, hl, LayerType::Tanh, "HL1", "AL2"),
        (hl, 2 * hl, LayerType::Tanh, "HL2", "AL3"),
        (2 * hl, hl, LayerType::Tanh, "HL3", "AL4"),
        (hl, hl / 2, LayerType::Tanh, "HL4", "AL5"),
        (hl / 2, hl / 2, LayerType::Tanh, "HL10", "AL11"),
        (hl / 2, hl / 2, LayerType::Tanh, "HL11", "AL12"),
        (hl / 2, input, LayerType::Sigmoid, "HL12", "Output"),
    ];

    for layer in layers {
        nn.add_linear(layer.0, layer.1, layer.3, distribution);
        nn.add_activation(layer.2, layer.4);
    }
    nn
}
