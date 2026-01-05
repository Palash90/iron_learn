use crate::nn::loss_functions::BinaryCrossEntropy;
use crate::nn::{DistributionType, LayerType};
use crate::NeuralNetBuilder;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufRead, BufReader};

use std::thread;
use std::time::Duration;

use crate::Tensor;
use crate::tensor::math::TensorMath;
use crate::numeric::FloatingPoint;

use crate::examples::contexts::GLOBAL_CONTEXT;

pub fn run_bigram_generator<T, D>() -> Result<(), String>
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
        let full_name = format!(".{}.", name);
        let chars_vec: Vec<char> = full_name.chars().collect();
        for window in chars_vec.windows(2) {
            let ix1 = stoi[&window[0]];
            let ix2 = stoi[&window[1]];

            // Create One-Hot for Input
            let mut oh = vec![D::zero(); vocab_size as usize];
            oh[ix1] = D::one();
            inputs.extend(oh);

            // Target (Simplified: one-hot for MSE)
            let mut target_oh = vec![D::zero(); vocab_size as usize];
            target_oh[ix2] = D::one();
            targets.extend(target_oh);
        }
    }

    let num_samples = (inputs.len() as u32) / vocab_size;
    let x_train = T::new(vec![num_samples, vocab_size], inputs)?;
    let y_train = T::new(vec![num_samples, vocab_size], targets)?;

    let mut nn = define_neural_net(hidden_length, vocab_size, &DistributionType::Xavier)
        .build(Box::new(BinaryCrossEntropy), &"BigramNet".to_string());

    // 4. Training (Small number of epochs for demo)
    nn.fit(
        &x_train,
        &y_train,
        e as usize,
        0,
        lr,
        true,
        |epoch, loss, _, _| {
            println!("\tEpoch {epoch}: Loss (BCE) = {loss:.8}");

            if sleep_time > 0 && epoch != 0 {
                println!("Taking a nap");
                thread::sleep(Duration::from_secs(5));
                println!("Awake again");
            }
        },
        1000,
    )?;

    // 5. Generation logic
    println!("\nGenerated Names:");
    for _ in 0..5 {
        let mut current_char = '.';
        let mut name = String::new();

        loop {
            let mut input_vec = vec![D::zero(); vocab_size as usize];
            input_vec[stoi[&current_char]] = D::one();
            let input_tensor = T::new(vec![1, vocab_size], input_vec)?;

            let preds = nn.predict(&input_tensor)?;
            let data = preds.get_data();

            // Sample: Find index with max value (Greedy)
            let next_ix = data
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.f64().partial_cmp(&b.f64()).unwrap())
                .map(|(i, _)| i)
                .unwrap();

            current_char = itos[&next_ix];
            if current_char == '.' {
                break;
            }
            name.push(current_char);
            if name.len() > 20 {
                break;
            } // Safety break
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

    let layers = [
        (input, hl, LayerType::Tanh, "Input", "AL 1"),
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
