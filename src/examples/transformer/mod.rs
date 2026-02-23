use rand::Rng;

use crate::examples::contexts::GLOBAL_CONTEXT;
use crate::examples::read_file::deserialize_model;
use crate::nn::layers::DistributionType;
use crate::nn::loss_functions::LossFunctionType;
use crate::nn::types::{TrainingConfig, TrainingHook};
use crate::numeric::FloatingPoint;
use crate::one_hot::one_hot_encode;
use crate::tensor::math::TensorMath;
use crate::utils::get_current_lr;
use crate::{NeuralNet, NeuralNetBuilder, Tensor};
use std::io::Write;
use std::thread::sleep;
use std::time::{Duration, Instant};

mod tokenizer;
mod vocabulary;

/// Represents a set of sequences and their corresponding target labels.
type Dataset = (Vec<Vec<u32>>, Vec<u32>);

/// A combined structure containing both Training and Validation data.
type SplitData = (Dataset, Dataset);

fn sample_with_temperature(probs: Vec<f32>, temp: f32) -> usize {
    let mut rng = rand::rng();
    let eps = 1e-10f32;

    // Softmax with temperature: exp(log(p) / T)
    let rescaled: Vec<f32> = probs.iter()
        .map(|&p| ((p + eps).ln() / temp).exp())
        .collect();

    let sum: f32 = rescaled.iter().sum();
    if sum <= 0.0 { return 0; }

    let r: f32 = rng.random_range(0.0..sum);
    let mut acc = 0.0;
    for (i, &p) in rescaled.iter().enumerate() {
        acc += p;
        if acc >= r { return i; }
    }
    probs.len() - 1
}

pub fn run_transformer_generator<T, D>() -> Result<(), String>
where
    T: crate::tensor::Tensor<D> + crate::tensor::math::TensorMath<D, MathOutput = T> + 'static,
    D: crate::numeric::FloatingPoint + 'static,
{
    let head_dim = 4;
    let num_heads = 8;

    let context = GLOBAL_CONTEXT
        .get()
        .ok_or("GLOBAL_CONTEXT not initialized")?;
    let sequence_length = context.n_gram_size as usize; // Context window

    let (vocab, lines) = vocabulary::build_vocabulary(&context.data_path)?;
    println!("Vocab Size: {}", vocab.vocab_size);

    let ((x_train, y_train), (x_val, y_val)): SplitData =
        prep_data(sequence_length, &vocab, lines)?;

    let mut nn: NeuralNet<T, D> =
        build_model(context, sequence_length, &vocab, head_dim, num_heads);

    if !context.predict_only {
        train(
            context,
            (x_train, y_train),
            (x_val, y_val),
            &mut nn,
            vocab.vocab_size,
            sequence_length,
        )?;
    }

    generate_sequence(sequence_length, vocab, nn)?;

    Ok(())
}

// In mod.rs
fn train<T, D>(
    context: &super::contexts::AppContext,
    train_data: (Vec<Vec<u32>>, Vec<u32>),
    val_data: (Vec<Vec<u32>>, Vec<u32>),
    nn: &mut NeuralNet<T, D>,
    vocab_size: u32,
    sequence_length: usize,
) -> Result<(), String>
where
    T: Tensor<D> + TensorMath<D, MathOutput = T> + 'static,
    D: FloatingPoint + 'static,
{
    let (x_train_raw, y_train_raw) = train_data;
    let (x_val_raw, y_val_raw) = val_data;
    let batch_size = 32; // Lowered from 64 to ensure safety on 4GB VRAM

    let start_epoch = if context.restore {
        nn.get_current_epoch()
    } else {
        0
    };
    let total_epochs = context.epochs as usize;

    for epoch in start_epoch..total_epochs {
        println!("\nStarting Epoch {}/{}", epoch + 1, total_epochs);
        let epoch_start_time = Instant::now();

        let lr: D = get_current_lr(
            D::from_f64(context.learning_rate),
            context.lr_adjust,
            D::from_f64(1e-6),
            total_epochs,
            epoch,
        );

        for i in (0..y_train_raw.len()).step_by(batch_size) {
            let end = (i + batch_size).min(y_train_raw.len());
            if end - i < batch_size {
                break;
            }

            // Create ONLY the current batch on the GPU
            let x_batch_vec: Vec<D> = x_train_raw[i..end]
                .iter()
                .flatten()
                .map(|&v| D::from_u32(v))
                .collect();
            let x_batch = T::new(vec![batch_size as u32, sequence_length as u32], x_batch_vec)
                .map_err(|e| e.to_string())?;
            let y_batch = one_hot_encode::<T, D>(&y_train_raw[i..end], vocab_size, 1)?;

            // Use a single validation batch to keep memory low
            let x_val_batch_vec: Vec<D> = x_val_raw[0..batch_size]
                .iter()
                .flatten()
                .map(|&v| D::from_u32(v))
                .collect();
            let x_val = T::new(
                vec![batch_size as u32, sequence_length as u32],
                x_val_batch_vec,
            )
            .map_err(|e| e.to_string())?;
            let y_val = one_hot_encode::<T, D>(&y_val_raw[0..batch_size], vocab_size, 1)?;

            nn.fit(
                &x_batch,
                &y_batch,
                &x_val,
                &y_val,
                TrainingConfig {
                    epochs: 1,
                    epoch_offset: 0,
                    base_lr: lr,
                    lr_adjustment: false, // Set to false since we're manually adjusting LR with get_current_lr
                    weight_normalization: false,
                },
                TrainingHook::new(1, |_, err, err_val, _, _| {
                    print!(
                        "\tBatch: {}/{}, Loss: {err:.4} , Val Loss : {err_val:.4}",
                        i / batch_size + 1,
                        y_train_raw.len() / batch_size
                    );

                    std::thread::sleep(Duration::from_millis(context.sleep_time));
                }),
            )?;
        }

        nn.set_current_epoch(epoch + 1);
        println!();
        nn.save_model(&context.weights_path);
        println!(
            "Epoch {} completed in {:.2} seconds",
            epoch + 1,
            epoch_start_time.elapsed().as_secs_f32()
        );
    }
    Ok(())
}

fn generate_sequence<T, D>(
    sequence_length: usize,
    vocab: vocabulary::Vocabulary,
    mut nn: NeuralNet<T, D>,
) -> Result<(), String>
where
    T: crate::tensor::Tensor<D> + crate::tensor::math::TensorMath<D, MathOutput = T> + 'static,
    D: crate::numeric::FloatingPoint + 'static,
{
    let context = GLOBAL_CONTEXT.get().ok_or("GLOBAL_CONTEXT not initialized")?;
    let seed_text = &context.n_gram_seed; 
    
    println!("\nGenerating text starting with seed: \"{}\"", seed_text);

    // 1. Initialize window with <PAD>
    let mut current_window_indices = vec![vocab.stoi["<PAD>"] as u32; sequence_length];

    // 2. Tokenize the seed and slide it into the window
    // We use the same grapheme tokenizer used in training
    let seed_tokens = tokenizer::tokenize_graphemes(seed_text);
    for token in seed_tokens {
        if let Some(&idx) = vocab.stoi.get(&token) {
            current_window_indices.remove(0);
            current_window_indices.push(idx as u32);
        }
    }

    let mut generated_words = Vec::new();
    
    loop {
        let input_tensor = T::new(
            vec![1, sequence_length as u32],
            current_window_indices
                .iter()
                .map(|&i| D::from_u32(i))
                .collect(),
        )
        .map_err(|e| format!("Failed to create input tensor: {}", e))?;

        let prediction = nn.predict(&input_tensor)?;
        let mut logits = prediction.get_data();

        // --- FIXED REPETITION PENALTY ---
        // We subtract from the raw logit. If we multiply a negative number 
        // by 0.5, it becomes LARGER (closer to zero), making it MORE likely.
        // Subtraction is a safer way to "push down" specific indices.
        let penalty_value = D::from_f64(2.0); 
        for word_str in &generated_words {
            if let Some(&historical_idx) = vocab.stoi.get(word_str) {
                logits[historical_idx] = logits[historical_idx] - penalty_value;
            }
        }

        // 3. Apply Temperature Sampling
        let temp_f64 = context.temparature.max(0.1); // Use context temp
        let temperature = D::from_f64(temp_f64);

        let adjusted_probs: Vec<f32> = logits
            .iter()
            .map(|&p| {
                // Softmax: exp(logit / T)
                (p / temperature).exp().f32()
            })
            .collect();

        // --- TOP-K FILTER ---
        let k = 5; 
        let mut indexed: Vec<(usize, f32)> = adjusted_probs
            .iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut filtered_probs = vec![0.0f32; vocab.vocab_size as usize];
        for i in indexed.iter().take(k) {
            let (idx, val) = *i;
            filtered_probs[idx] = val;
        }

        let next_idx = sample_with_temperature(filtered_probs, temp_f64 as f32);
        let next_word = vocab.itos.get(&next_idx).ok_or("Index not in vocab")?;

        if next_word == "<END>" || generated_words.len() > 200 {
            break;
        }

        generated_words.push(next_word.clone());
        
        // Slide the window for the next prediction
        current_window_indices.remove(0);
        current_window_indices.push(next_idx as u32);

        // Print the seed + generated text
        print!("\r{}{}", seed_text, generated_words.join(""));
        let _ = std::io::stdout().flush();
        sleep(Duration::from_millis(50));
    }
    println!();
    Ok(())
}
fn build_model<T, D>(
    context: &super::contexts::AppContext,
    sequence_length: usize,
    vocab: &vocabulary::Vocabulary,
    head_dim: u32,
    num_heads: u32,
) -> NeuralNet<T, D>
where
    T: crate::tensor::Tensor<D> + crate::tensor::math::TensorMath<D, MathOutput = T> + 'static,
    D: crate::numeric::FloatingPoint + 'static,
{
    // --- 3. BUILD MODEL ---
    let nn = if !context.restore {
        let mut builder = NeuralNetBuilder::new();

        // Adaptive sizing based on vocabulary size
        // Larger vocabularies need smaller embeddings to fit in 4GB VRAM
        let (head_dim, num_heads) = (head_dim, num_heads);

        let embed_dim = head_dim * num_heads; // Per-token embedding dimension
        let total_embed_dim = embed_dim * sequence_length as u32; // Flattened

        println!(
            "Model configuration: embed_dim={}, num_heads={}, head_dim={}, seq_len={}",
            embed_dim, num_heads, head_dim, sequence_length
        );

        builder.add_embedding(
            vocab.vocab_size,
            sequence_length as u32,
            embed_dim,
            "embedding",
        );
        // Add transformer with per-token embeddings divided into heads
        builder.add_transformer_with_seq(
            embed_dim,
            sequence_length as u32,
            num_heads,
            "transformer",
            &DistributionType::Xavier,
        );
        builder.add_linear(
            total_embed_dim,
            vocab.vocab_size,
            "head",
            &DistributionType::Xavier,
        );

        builder.build(LossFunctionType::CategoricalCrossEntropy, "Transformer")
    } else {
        // First, load the model to check vocab size BEFORE building
        let model_for_check = deserialize_model::<D>(&context.weights_path).unwrap();
        let saved_vocab_size = model_for_check.layers[2].shape[1]; // Head layer output size = vocab (shape is [input, output])

        if saved_vocab_size != vocab.vocab_size {
            eprintln!("\n⚠️  WARNING: Vocabulary size mismatch!");
            eprintln!(
                "   Saved model was trained with vocab size: {}",
                saved_vocab_size
            );
            eprintln!("   Current data has vocab size: {}", vocab.vocab_size);
            eprintln!("   This will cause errors unless you use the same data file.\n");
            eprintln!("   Options:");
            eprintln!("   1. Use the original data file that matches this model");
            eprintln!("   2. Train a new model with -n <new_name> instead of -r (restore)");
            eprintln!("   3. Delete the old model and start fresh\n");

            eprintln!(
                "   → Recreating model with new vocabulary size (weights will be reinitialized)\n"
            );

            let mut builder = NeuralNetBuilder::new();

            let (head_dim, num_heads) = (head_dim, num_heads);

            let embed_dim = head_dim * num_heads;
            let total_embed_dim = embed_dim * sequence_length as u32;

            builder.add_embedding(
                vocab.vocab_size,
                sequence_length as u32,
                embed_dim,
                "embedding",
            );
            builder.add_transformer_with_seq(
                embed_dim,
                sequence_length as u32,
                num_heads,
                "transformer",
                &DistributionType::Xavier,
            );
            builder.add_linear(
                total_embed_dim,
                vocab.vocab_size,
                "head",
                &DistributionType::Xavier,
            );

            builder.build(LossFunctionType::CategoricalCrossEntropy, "Transformer")
        } else {
            // Vocab size matches, safe to restore
            let model = deserialize_model::<D>(&context.weights_path).unwrap();
            NeuralNetBuilder::<T, D>::build_from_model(model)
        }
    };
    nn
}

// In mod.rs
fn prep_data(
    sequence_length: usize,
    vocab: &vocabulary::Vocabulary,
    lines: Vec<String>,
) -> Result<SplitData, String> {
    let mut all_inputs: Vec<Vec<u32>> = Vec::new();
    let mut all_targets: Vec<u32> = Vec::new();

    for line in &lines {
        let mut tokens = tokenizer::tokenize_graphemes(line);
        if tokens.is_empty() {
            continue;
        }
        tokens.push("<END>".to_string());

        let mut window: Vec<u32> = vec![vocab.stoi["<PAD>"] as u32; sequence_length];

        for token in tokens {
            all_inputs.push(window.clone());
            let target_idx = *vocab.stoi.get(&token).unwrap();
            all_targets.push(target_idx as u32);

            window.remove(0);
            window.push(target_idx as u32);
        }
    }

    let train_ratio = 0.8;
    let train_size = (all_targets.len() as f64 * train_ratio) as usize;

    let (train_in, test_in) = all_inputs.split_at(train_size);
    let (train_tar, test_tar) = all_targets.split_at(train_size);

    Ok((
        (train_in.to_vec(), train_tar.to_vec()),
        (test_in.to_vec(), test_tar.to_vec()),
    ))
}
