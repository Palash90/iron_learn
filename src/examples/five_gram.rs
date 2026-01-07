use crate::nn::loss_functions::CategoricalCrossEntropy;
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
use crate::nn::types::TrainingConfig;
use crate::nn::types::TrainingHook;
use crate::NeuralNet;
use std::time::Instant;

use colored::*;

pub fn run_five_gram_generator<T, D>() -> Result<(), String>
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
    let weights_path = "model_outputs/".to_owned() + &name.to_owned() + "/" + weights_path;
    let lr_adjustment = GLOBAL_CONTEXT.get().unwrap().lr_adjust;
    let restore = GLOBAL_CONTEXT.get().unwrap().restore;
    let predict_only = GLOBAL_CONTEXT.get().unwrap().predict_only;
    let resize = GLOBAL_CONTEXT.get().unwrap().resize;
    let temparature = GLOBAL_CONTEXT.get().unwrap().temparature;

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

    let mut generation_set = HashSet::new();
    let names_set: HashSet<String> = lines.clone().into_iter().collect();
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
        let full_name = format!("....{}.", name);
        let chars_vec: Vec<char> = full_name.chars().collect();
        for window in chars_vec.windows(5) {
            let ix1 = stoi[&window[0]];
            let ix2 = stoi[&window[1]];
            let ix3 = stoi[&window[2]];
            let ix4 = stoi[&window[3]];
            let ix5 = stoi[&window[4]];

            // Create One-Hot for Input
            let mut oh = vec![D::zero(); (vocab_size * 4) as usize];
            oh[ix1] = D::one();
            oh[ix2 + vocab_size as usize] = D::one();
            oh[ix3 + ((vocab_size * 2) as usize)] = D::one();
            oh[ix4 + ((vocab_size * 3) as usize)] = D::one();
            inputs.extend(oh);

            // Target (Simplified: one-hot for MSE)
            let mut target_oh = vec![D::zero(); vocab_size as usize];
            target_oh[ix5] = D::one();
            targets.extend(target_oh);
        }
    }

    let num_samples = (inputs.len() as u32) / (vocab_size * 4);
    let x_train = T::new(vec![num_samples, vocab_size * 4], inputs)?;
    let y_train = T::new(vec![num_samples, vocab_size], targets)?;

    let loss_function_instance = Box::new(CategoricalCrossEntropy);

    let (l, epoch_offset, mut nn) = match !weights_path.is_empty() && restore {
        true => match deserialize_model::<D>(&weights_path) {
            Some(model) => (
                model.saved_lr,
                model.epoch,
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

    let config = TrainingConfig {
        epochs: e as usize,
        epoch_offset,
        base_lr: l,
        lr_adjustment,
    };

    let mut start_time = Instant::now();
    let mut last_epoch = 0;

    let mut generate_names = |nn: &mut NeuralNet<T, D>, num: usize, sleep: u64| {
        println!("\nGenerated Names:");
        let mut fresh_count = 0;

        for i in 0..num {
            let mut context = ('.', '.', '.', '.');
            let mut name = String::new();

            loop {
                let mut input_vec = vec![D::zero(); (vocab_size * 4) as usize];
                input_vec[stoi[&context.0]] = D::one();
                input_vec[stoi[&context.1] + vocab_size as usize] = D::one();
                input_vec[stoi[&context.2] + (vocab_size * 2) as usize] = D::one();
                input_vec[stoi[&context.3] + (vocab_size * 3) as usize] = D::one();

                let input_tensor = T::new(vec![1, vocab_size * 4], input_vec).unwrap();
                let preds = nn.predict(&input_tensor).unwrap();
                let data = preds.get_data();

                let mut weights: Vec<f64> = data
                    .iter()
                    .map(|val| (val.f64() / temparature).exp())
                    .collect();

                if name.len() > 5 {
                    weights[0] *= 2.0; // Double the chance of ending
                }
                if name.len() > 7 {
                    weights[0] *= 10.0; // Force an end
                }

                let total_weight: f64 = weights.iter().sum();

                let rng_seed = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos() as f64;
                let random_point = (rng_seed % 1_000_000.0 / 1_000_000.0) * total_weight;

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

                context = (context.1, context.2, context.3, next_char);

                if name.len() > 20 {
                    break;
                }
            }

            let is_fresh = !names_set.contains(&name);
            let duplicate_generation = generation_set.contains(&name);
            generation_set.insert(name.clone());

            if duplicate_generation {
                continue;
            }

            let status_symbol = if is_fresh { "✓".green() } else { "✗".red() };
            let status_text = if is_fresh { "NEW".green() } else { "OLD".red() };

            if is_fresh {
                fresh_count += 1;
            }

            let originality_pct = (fresh_count as f64 * 100.0) / ((i + 1) as f64);
            let originality_pct = format!("{:.2}%", originality_pct);

            println!(
                "{} {:<15} {:>10} | Innovation Rate: {}",
                status_symbol,
                name.bright_white(),
                status_text,
                originality_pct.cyan()
            );

            if sleep > 0 {
                thread::sleep(Duration::from_secs(sleep));
            }
        }
        println!();
    };

    let monitor = |epoch, loss: D, _, nn: &mut NeuralNet<T, D>| {
        let elapsed = start_time.elapsed();
        start_time = Instant::now();

        if loss.f64().is_nan() {
            panic!("Hit NaN");
        }

        println!("\tEpoch {epoch}: Loss (CCE) = {loss:.8}, {last_epoch} - {epoch} time elapsed: {elapsed:.2?}");
        last_epoch = epoch;
        nn.save_model(format!("{}_epoch_{}.json", &weights_path, &epoch.to_string()).as_str());
        //generate_names(nn, 10, 0);

        if sleep_time > 0 && epoch != 0 {
            println!("Taking a nap");
            thread::sleep(Duration::from_secs(sleep_time));
            println!("Awake again");
        }
    };
    let hook_config = TrainingHook::new(1000, monitor);

    if !predict_only {
        nn.fit(&x_train, &y_train, config, hook_config)?;
    } else {
        println!("Skipping training...");
    }

    println!("Final Generation");
    generate_names(&mut nn, resize as usize, sleep_time);

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

    let input_size = input * 4;

    let layers = [
        (input_size, hl, LayerType::ReLU, "Input", "AL 1"),
        (hl, hl, LayerType::ReLU, "HL1", "AL2"),
        (hl, 2 * hl, LayerType::ReLU, "HL2", "AL3"),
        //    (2 * hl, hl, LayerType::ReLU, "HL3", "AL4"),
        (2 * hl, hl / 2, LayerType::ReLU, "HL4", "AL5"),
        (hl / 2, hl / 4, LayerType::ReLU, "HL10", "AL11"),
        //  (hl / 2, hl / 2, LayerType::ReLU, "HL11", "AL12"),
        (hl / 4, input, LayerType::Softmax, "HL12", "Output"),
    ];

    for layer in layers {
        nn.add_linear(layer.0, layer.1, layer.3, distribution);
        nn.add_activation(layer.2, layer.4);
    }
    nn
}
