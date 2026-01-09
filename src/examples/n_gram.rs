use crate::nn::loss_functions::CategoricalCrossEntropy;
use crate::nn::{DistributionType, LayerType};
use crate::NeuralNetBuilder;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};

use std::thread;
use std::time::Duration;

use crate::numeric::FloatingPoint;
use crate::tensor::math::TensorMath;
use crate::Tensor;

use crate::examples::contexts::GLOBAL_CONTEXT;
use crate::examples::read_file::deserialize_model;
use crate::nn::types::TrainingConfig;
use crate::nn::types::TrainingHook;
use crate::one_hot::one_hot_encode;
use crate::NeuralNet;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

use rand::seq::SliceRandom;
use std::time::Instant;

use colored::*;

#[derive(Debug, Serialize, Deserialize)]
struct NGramMetadata {
    vocab_size: u32,
    stoi: HashMap<char, usize>,
    itos: HashMap<usize, char>,
    multiplier: u32,
}

pub fn run_n_gram_generator<T, D>() -> Result<(), String>
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
    let lr_adjustment = GLOBAL_CONTEXT.get().unwrap().lr_adjust;
    let restore = GLOBAL_CONTEXT.get().unwrap().restore;
    let data_path = &GLOBAL_CONTEXT.get().unwrap().data_path;
    let predict_only = GLOBAL_CONTEXT.get().unwrap().predict_only;
    let resize = GLOBAL_CONTEXT.get().unwrap().resize;
    let temparature = GLOBAL_CONTEXT.get().unwrap().temparature;
    let temparature = if temparature == 0.0 { 0.1 } else { temparature };
    let no_repeat = !GLOBAL_CONTEXT.get().unwrap().repeat;
    let n_gram_seed = &GLOBAL_CONTEXT.get().unwrap().n_gram_seed;
    let n_gram_size = GLOBAL_CONTEXT.get().unwrap().n_gram_size;
    let weights_path = &GLOBAL_CONTEXT.get().unwrap().weights_path;

    let metadata_file = "model_outputs/".to_owned() + &name.to_owned() + "/n_gram_metadata.json";

    let file = File::open(data_path).expect("File could not be opened");
    let reader = BufReader::new(file);

    let lines: Vec<String> = reader
        .lines()
        .map(|line| line.expect("Failed to read line").to_lowercase())
        .collect();

    let mut names = HashSet::new();
    names.extend(lines.iter());
    let mut names: Vec<&String> = names.into_iter().collect();
    println!("Unique names: {}", names.len());

    let mut generation_set = HashSet::new();
    let names_set: HashSet<String> = lines.clone().into_iter().collect();

    let metadata = match restore {
        false => {
            let mut chars: Vec<char> = names
                .iter()
                .flat_map(|s| s.chars())
                .collect::<HashSet<_>>()
                .into_iter()
                .collect();
            chars.sort();
            chars.insert(0, '.');

            let vocab_size = chars.len() as u32;

            let stoi: HashMap<char, usize> =
                chars.iter().enumerate().map(|(i, &c)| (c, i)).collect();
            let itos: HashMap<usize, char> =
                chars.iter().enumerate().map(|(i, &c)| (i, c)).collect();

            let multiplier: u32 = n_gram_size as u32 - 1;

            (stoi, itos, vocab_size, multiplier)
        }
        true => {
            let contents = match fs::read_to_string(&metadata_file) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("{}", &format!("Failed to read file: {}", e));
                    String::new()
                }
            };

            let data: NGramMetadata = match serde_json::from_str(&contents) {
                Ok(d) => d,
                Err(err) => {
                    panic!("N Gram Metadata could not be read: {}", err);
                }
            };
            (data.stoi, data.itos, data.vocab_size, data.multiplier)
        }
    };

    if !restore {
        let metadata = NGramMetadata {
            stoi: metadata.0.clone(),
            itos: metadata.1.clone(),
            vocab_size: metadata.2,
            multiplier: metadata.3,
        };

        let path = Path::new(&metadata_file);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap(); // Creates all directories if they don't exist
        }

        serde_json::to_writer_pretty(File::create(&metadata_file).unwrap(), &metadata).unwrap();
    }

    let (stoi, itos, vocab_size, multiplier) = metadata;
    println!("Vocabulary Size {}: {:?}", vocab_size, stoi);

    let loss_function_instance = Box::new(CategoricalCrossEntropy);

    let (l, epoch_offset, mut nn) = match !weights_path.is_empty() && restore {
        true => match deserialize_model::<D>(weights_path) {
            Some(model) => (
                model.saved_lr,
                model.epoch,
                NeuralNetBuilder::build_from_model(model, loss_function_instance),
            ),
            None => (
                lr,
                0,
                define_neural_net::<T, D>(hidden_length, vocab_size, distribution, multiplier)
                    .build(loss_function_instance, name),
            ),
        },
        false => (
            lr,
            0,
            define_neural_net::<T, D>(hidden_length, vocab_size, distribution, multiplier)
                .build(loss_function_instance, name),
        ),
    };

    if !predict_only {
        let build_xy = |name_list: &[&String]| -> (Vec<u32>, Vec<u32>) {
            let mut inputs = Vec::new();
            let mut targets = Vec::new();
            let pad_str = ".".repeat(multiplier as usize);

            for name in name_list {
                let full_name = format!("{}{}.", pad_str, name);
                let chars_vec: Vec<char> = full_name.chars().collect();

                for window in chars_vec.windows(multiplier as usize + 1) {
                    for i in 0..multiplier {
                        inputs.push(stoi[&window[i as usize]] as u32);
                    }
                    targets.push(stoi[&window[multiplier as usize]] as u32);
                }
            }
            (inputs, targets)
        };

        let mut rng = rand::rng();
        names.shuffle(&mut rng);

        let total_names = names.len();
        let train_size = (total_names as f64 * 0.8) as usize; // 80% Training

        let train_names = &names[0..train_size];
        let val_names = &names[train_size..];

        let (train_in, train_tar) = build_xy(train_names);
        let x_train = one_hot_encode(&train_in, vocab_size, multiplier)?;
        let y_train = one_hot_encode::<T, D>(&train_tar, vocab_size, 1)?;

        // Generate Validation Data
        let (val_in, val_tar) = build_xy(val_names);
        let x_val = one_hot_encode(&val_in, vocab_size, multiplier)?;
        let y_val = one_hot_encode::<T, D>(&val_tar, vocab_size, 1)?;

        // Add Label Smoothing
        let epsilon = D::from_f64(0.1); // The "smoothing" factor
        let num_classes = D::from_u32(vocab_size);

        let y_train_data = y_train.get_data();
        let mut smooth_data = vec![];

        for val in y_train_data {
            // Standard one-hot is [0, 1, 0]
            // Smoothed becomes [0.003, 0.903, 0.003]
            smooth_data.push(val * (D::one() - epsilon) + (epsilon / num_classes));
        }

        let y_train = T::new(y_train.get_shape().to_vec(), smooth_data)?;

        let mut start_time = Instant::now();
        let mut last_epoch = 0;

        let monitor = |epoch, loss: D, val_loss: D, _, nn: &mut NeuralNet<T, D>| {
            let elapsed = start_time.elapsed();
            start_time = Instant::now();

            if loss.f64().is_nan() {
                panic!("Hit NaN");
            }

            println!("\tEpoch {epoch}: Loss (CCE) = {loss:.4}, Val Loss (CCE) = {val_loss:.4}, {last_epoch} - {epoch} time elapsed: {elapsed:.2?}");

            last_epoch = epoch;
            nn.save_model(weights_path);

            if sleep_time > 0 && epoch != 0 {
                println!("Taking a nap");
                thread::sleep(Duration::from_secs(5));
                println!("Awake again");
            }
        };
        let config = TrainingConfig {
            epochs: e as usize,
            epoch_offset,
            base_lr: l,
            lr_adjustment,
        };

        let hook_config = TrainingHook::new(1000, monitor);

        nn.fit(&x_train, &y_train, &x_val, &y_val, config, hook_config)?;
    }

    println!("\nGenerated Names:");
    let mut fresh_count = 0;
    let mut new_names = 0;

    for i in 0..resize {
        // This is the full seed, it can be bigger than the multiplier
        let full_seed = n_gram_seed.clone().to_lowercase();
        let multiplier = multiplier as usize;

        let seed_chars: Vec<char> = full_seed.chars().collect();

        let seed_len = seed_chars.len();

        let mut name = if seed_len > multiplier {
            seed_chars[seed_len - multiplier..]
                .iter()
                .collect::<String>()
        } else {
            full_seed.clone()
        };

        let mut c_vec = vec!['.'; multiplier];
        let name_chars: Vec<char> = name.chars().collect();

        for (i, &ch) in name_chars.iter().rev().enumerate() {
            if i < multiplier {
                c_vec[multiplier - 1 - i] = ch;
            }
        }

        loop {
            let mut input_vec = vec![D::zero(); vocab_size as usize * multiplier];

            for (pos, &ch) in c_vec.iter().enumerate() {
                let offset = pos * vocab_size as usize;
                input_vec[offset + stoi[&ch]] = D::one();
            }

            let input_tensor = T::new(vec![1, vocab_size * multiplier as u32], input_vec)?;
            let preds = nn.predict(&input_tensor)?;
            let data = preds.get_data();

            let mut weights: Vec<f64> = data
                .iter()
                .map(|val| (val.f64() / temparature).exp())
                .collect();

            if name.len() < 3 {
                weights[0] = 0.0;
            }

            if name.len() > multiplier {
                weights[0] *= 2.0;
            }
            if name.len() > (3 * multiplier) {
                weights[0] *= 10.0;
            }

            let total_weight: f64 = weights.iter().sum();

            if total_weight == 0.0 {
                break;
            }

            let chars_vec: Vec<char> = name.chars().collect();
            if chars_vec.len() >= 2 {
                let last = chars_vec[chars_vec.len() - 1];
                let second_last = chars_vec[chars_vec.len() - 2];

                if last == second_last {
                    // Find index of the repeating character and kill its probability
                    if let Some(&idx) = stoi.get(&last) {
                        weights[idx] = 0.0;
                    }
                }
            }

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
                break;
            }

            name.push(next_char);

            c_vec.remove(0);
            c_vec.push(next_char);

            if name.len() > 20 {
                break;
            }
        }

        let name = name.trim().to_string();
        let is_original = names_set.contains(&name);
        let is_duplicate = generation_set.contains(&name);
        generation_set.insert(name.clone());

        let status_symbol = if is_original {
            "✗".red()
        } else if is_duplicate {
            "!".yellow()
        } else {
            "✓".green()
        };

        let status_text = if is_original {
            "OLD".red()
        } else if is_duplicate {
            "DUP".yellow()
        } else {
            "NEW".green()
        };

        if !is_original && !is_duplicate {
            fresh_count += 1;
        }

        let originality_pct = (fresh_count as f64 * 100.0) / ((i + 1) as f64);
        let originality_pct = format!("{:.2}%", originality_pct);

        if no_repeat && (is_original || is_duplicate) {
            continue;
        }

        if !is_pronounceable(&name) && name.len() > 3 {
            continue;
        }

        println!(
            "{} '{:<15}' {:>10} | Innovation Rate: {}",
            status_symbol,
            name.bright_white(),
            status_text,
            originality_pct.cyan()
        );
        new_names += 1;
    }

    println!("Total new names generated: {new_names}");
    Ok(())
}

fn is_pronounceable(name: &str) -> bool {
    let len = name.len();
    if len < 2 {
        return false;
    } // Too short

    let vowels = ['a', 'e', 'i', 'o', 'u', 'y'];
    let v_count = name.chars().filter(|c| vowels.contains(c)).count();
    let ratio = v_count as f32 / len as f32;

    // 1. Dynamic Ratio: Short names need more flexibility than long ones
    let (min_r, max_r) = if len <= 3 { (0.25, 0.75) } else { (0.30, 0.65) };
    if ratio < min_r || ratio > max_r {
        return false;
    }

    let chars: Vec<char> = name.chars().collect();

    // 2. Window-based Phonetic Checks
    // Check for triple consonants AND triple vowels
    let has_bad_cluster = chars.windows(3).any(|w| {
        let all_vowels = w.iter().all(|c| vowels.contains(c));
        let all_consonants = w.iter().all(|c| !vowels.contains(c));
        all_vowels || all_consonants
    });

    if has_bad_cluster {
        return false;
    }

    // 3. Start/End Sanity
    // Names usually don't end in certain consonants like 'q' or 'j'
    let last_char = chars[len - 1];
    if ['q', 'j', 'x', 'v'].contains(&last_char) && len > 3 {
        return false;
    }

    true
}

fn define_neural_net<T, D>(
    hl: u32,
    input: u32,
    distribution: &DistributionType,
    multiplier: u32,
) -> NeuralNetBuilder<T, D>
where
    T: Tensor<D> + TensorMath<D, MathOutput = T> + 'static,
    D: FloatingPoint + 'static,
{
    let mut nn = NeuralNetBuilder::<T, D>::new();

    let input_size = input * multiplier;

    let layers = [
        (input_size, hl, LayerType::ReLU, "Input", "AL 1"),
        // (hl, hl, LayerType::ReLU, "HL2", "AL3"),
        //  (2 * hl, hl, LayerType::ReLU, "HL4", "AL5"),
        (hl, hl / 2, LayerType::ReLU, "HL10", "AL11"),
        (hl / 2, hl, LayerType::ReLU, "HL10", "AL11"),
        (hl, input, LayerType::Softmax, "HL12", "Output"),
    ];

    for layer in layers {
        nn.add_linear(layer.0, layer.1, layer.3, distribution);
        nn.add_activation(layer.2, layer.4);
    }
    nn
}
