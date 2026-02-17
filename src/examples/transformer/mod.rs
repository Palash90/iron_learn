use rand::Rng;

use crate::examples::contexts::GLOBAL_CONTEXT;
use crate::examples::read_file::deserialize_model;
use crate::nn::layers::DistributionType;
use crate::nn::loss_functions::LossFunctionType;
use crate::nn::types::{TrainingConfig, TrainingHook};
use crate::numeric::FloatingPoint;
use crate::one_hot::one_hot_encode;
use crate::tensor::math::TensorMath;
use crate::{NeuralNet, NeuralNetBuilder, Tensor};
use std::io::Write;
use std::thread::sleep;
use std::time::{Duration, Instant};

mod tokenizer;
mod vocabulary;

fn sample_with_temperature(probs: Vec<f32>, temp: f32) -> usize {
    let mut rng = rand::rng();

    // 1. Rescale probabilities using temperature
    let rescaled: Vec<f32> = probs.iter().map(|p| (p.ln() / temp).exp()).collect();

    let sum: f32 = rescaled.iter().sum();
    let r: f32 = rng.random_range(0.0..sum);

    // 2. Weighted Random Selection
    let mut acc = 0.0;
    for (i, p) in rescaled.iter().enumerate() {
        acc += p;
        if acc >= r {
            return i;
        }
    }
    probs.len() - 1
}

pub fn run_transformer_generator<T, D>() -> Result<(), String>
where
    T: crate::tensor::Tensor<D> + crate::tensor::math::TensorMath<D, MathOutput = T> + 'static,
    D: crate::numeric::FloatingPoint + 'static,
{
    let head_dim = 16;
    let num_heads = 4;

    let context = GLOBAL_CONTEXT
        .get()
        .ok_or("GLOBAL_CONTEXT not initialized")?;
    let sequence_length = context.n_gram_size as usize; // Context window

    let (vocab, lines) = vocabulary::build_vocabulary(&context.data_path)?;
    println!("Vocab Size: {}", vocab.vocab_size);

    let ((x_train, y_train), (x_val, y_val)): ((Vec<T>, Vec<T>), (Vec<T>, Vec<T>)) = prep_data(sequence_length, &vocab, lines)?;

    let mut nn: NeuralNet<T, D> = build_model(context, sequence_length, &vocab, head_dim, num_heads);

    train(context, x_train, y_train, x_val, y_val, &mut nn)?;

    generate_sequence(sequence_length, vocab, nn)?;

    Ok(())
}

fn train<T, D>(
    context: &super::contexts::AppContext,
    x_train: Vec<T>,
    y_train: Vec<T>,
    x_val: Vec<T>,
    y_val: Vec<T>,
    nn: &mut NeuralNet<T, D>,
) -> Result<(), String>
where
    T: Tensor<D> + TensorMath<D, MathOutput = T> + 'static,
    D: FloatingPoint + 'static,
{
    let start_epoch = if context.restore { nn.get_current_epoch() } else { 0 };
    let total_epochs = context.epochs as usize;

    if GLOBAL_CONTEXT.get().unwrap().predict_only { return Ok(()); }

    for epoch in start_epoch..total_epochs {
        println!("Starting Epoch {}/{}", epoch + 1, total_epochs);
        
        // Track cumulative loss for the epoch
        let mut epoch_loss = 0.0;

        for i in 0..x_train.len() {
            let val_idx = i % x_val.len();

            let config = TrainingConfig {
                epochs: 1, 
                epoch_offset: 0, // This tells the NN which epoch we are on
                base_lr: D::from_f64(context.learning_rate),
                lr_adjustment: context.lr_adjust,
                weight_normalization: false,
            };

            nn.fit(
                &x_train[i],
                &y_train[i],
                &x_val[val_idx],
                &y_val[val_idx],
                config,
                TrainingHook::new(
                    usize::MAX, // Disable internal fit logging to use our own
                    |_, l, _, _, _| { /* empty */ },
                ),
            )?;
            
            // Log manually every 20 batches
            if i % 20 == 0 || i == x_train.len() - 1 {
                println!("Epoch {} | Batch {}/{} | Processing...", epoch + 1, i, x_train.len());
            }
        }

        println!("Completed Epoch {}. Saving checkpoint...", epoch + 1);
        nn.save_model(&context.weights_path);
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
    println!("\nGenerating text...");
    let mut current_window_indices = vec![vocab.stoi["<PAD>"] as u32; sequence_length];
    let mut generated_words = Vec::new();
    let penalty_factor = 1.2;
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
        let mut probs = prediction.get_data();

        for word_str in &generated_words {
            if let Some(&historical_idx) = vocab.stoi.get(word_str) {
                let idx = historical_idx as usize;
                // Setting the logit to a massive negative number
                // effectively "deletes" the word from the current options.
                probs[idx] = D::from_f64(-100.0);
            }
        }

        // for word_str in &generated_words {
        //     if let Some(&historical_idx) = vocab.stoi.get(word_str) {
        //         let idx = historical_idx as usize;
        //         let global_penalty = D::from_f64(2.0); // Increased from 1.2 to 2.0 to break the loop

        //         if probs[idx] > D::from_f64(0.0) {
        //             probs[idx] = probs[idx] / global_penalty;
        //         } else {
        //             probs[idx] = probs[idx] * global_penalty;
        //         }
        //     }
        // }

        // 3. Apply Temperature Sampling
        let temp_f64 = 1.0; // Lowered slightly since penalty adds its own diversity
        let temperature = D::from_f64(temp_f64);

        let adjusted_probs: Vec<f32> = probs
            .iter()
            .map(|&p| {
                let epsilon = D::from_f64(1e-10);
                let safe_p = if p < epsilon { epsilon } else { p };
                (safe_p.ln() / temperature).exp().f32()
            })
            .collect();

        // --- TOP-K FILTER ---
        let k = 5; // Only consider the top 20 most likely words
        let mut indexed: Vec<(usize, f32)> = adjusted_probs
            .iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut filtered_probs = vec![0.0f32; vocab.vocab_size as usize];
        for i in 0..k {
            let (idx, val) = indexed[i];
            filtered_probs[idx] = val;
        }

        let next_idx = sample_with_temperature(filtered_probs, temp_f64 as f32);

        let next_word = vocab.itos.get(&next_idx).ok_or("Index not in vocab")?;

        if next_word == "<END>" || generated_words.len() > 100 {
            break;
        }

        generated_words.push(next_word.clone());
        current_window_indices.remove(0);
        current_window_indices.push(next_idx as u32);

        print!("\r{}", generated_words.join(" "));
        let _ = std::io::stdout().flush();
        sleep(Duration::from_millis(100));
    }
    println!("Result: {}", generated_words.join(" "));
    Ok(())
}

fn build_model<T, D>(
    context: &super::contexts::AppContext,
    sequence_length: usize,
    vocab: &vocabulary::Vocabulary,
    head_dim: u32,
    num_heads: u32
) -> NeuralNet<T, D>
where
    T: crate::tensor::Tensor<D> + crate::tensor::math::TensorMath<D, MathOutput = T> + 'static,
    D: crate::numeric::FloatingPoint + 'static,
{
    // --- 3. BUILD MODEL ---
    let mut nn = if !context.restore {
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

        let nn = builder.build(LossFunctionType::CategoricalCrossEntropy, "Transformer");

        nn
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

fn prep_data<T, D>(
    sequence_length: usize,
    vocab: &vocabulary::Vocabulary,
    lines: Vec<String>,
) -> Result<((Vec<T>, Vec<T>), (Vec<T>, Vec<T>)), String> // Returns ((train_x, train_y), (test_x, test_y))
where
    T: Tensor<D> + TensorMath<D, MathOutput = T> + 'static,
    D: FloatingPoint + 'static,
{
    let batch_size: usize = 64;
    let train_ratio = 0.8; // 80% for training
    
    // 1. Collect all individual examples first
    let mut all_inputs: Vec<Vec<u32>> = Vec::new();
    let mut all_targets: Vec<u32> = Vec::new();

    for line in &lines {
        let mut tokens = tokenizer::tokenize(line);
        if tokens.is_empty() { continue; }
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

    // 2. Split the raw data
    let total_samples = all_targets.len();
    let train_size = (total_samples as f64 * train_ratio) as usize;

    // Split vectors
    let (train_in, test_in) = all_inputs.split_at(train_size);
    let (train_tar, test_tar) = all_targets.split_at(train_size);

    // 3. Helper to batch tensors
    let create_batches = |inputs: &[Vec<u32>], targets: &[u32]| -> Result<(Vec<T>, Vec<T>), String> {
        let mut x_batches = Vec::new();
        let mut y_batches = Vec::new();
        
        let num_batches = targets.len() / batch_size;
        
        for i in 0..num_batches {
            let start = i * batch_size;
            let end = start + batch_size;
            
            // Flatten the input windows for this batch
            let batch_in_flat: Vec<D> = inputs[start..end]
                .iter()
                .flatten()
                .map(|&val| D::from_u32(val))
                .collect();

            let x_batch = T::new(
                vec![batch_size as u32, sequence_length as u32],
                batch_in_flat,
            ).map_err(|e| e.to_string())?;

            let y_batch = one_hot_encode::<T, D>(&targets[start..end], vocab.vocab_size, 1)?;

            x_batches.push(x_batch);
            y_batches.push(y_batch);
        }
        Ok((x_batches, y_batches))
    };

    let train_data = create_batches(train_in, train_tar)?;
    let test_data = create_batches(test_in, test_tar)?;

    Ok((train_data, test_data))
}