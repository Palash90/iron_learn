use rand::Rng;

use crate::examples::contexts::GLOBAL_CONTEXT;
use crate::examples::read_file::deserialize_model;
use crate::nn::layers::DistributionType;
use crate::nn::loss_functions::LossFunctionType;
use crate::nn::types::{TrainingConfig, TrainingHook};
use crate::one_hot::one_hot_encode;
use crate::{NeuralNet, NeuralNetBuilder};
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
    let context = GLOBAL_CONTEXT
        .get()
        .ok_or("GLOBAL_CONTEXT not initialized")?;
    let sequence_length = context.n_gram_size as usize; // Context window

    let (vocab, lines) = vocabulary::build_vocabulary(&context.data_path)?;
    println!("Vocab Size: {}", vocab.vocab_size);

    // --- 3. DATA PREP (Sliding Window) ---
    let mut inputs = Vec::new();
    let mut targets = Vec::new();

    // Initialize the window ONCE before we start processing lines
    let mut window = vec![vocab.stoi["<PAD>"] as u32; sequence_length];

    for line in &lines {
        let mut tokens = tokenizer::tokenize(line);
        if tokens.is_empty() {
            continue;
        }

        tokens.push("<END>".to_string());

        for token in tokens {
            inputs.extend_from_slice(&window);

            let target_idx = *vocab
                .stoi
                .get(&token)
                .unwrap_or_else(|| panic!("Token '{}' not found!", token));

            targets.push(target_idx as u32);

            // Slide window
            window.remove(0);
            window.push(target_idx as u32);
        }
    }

    let num_examples = targets.len() as u32;

    let x_train = T::new(
        vec![num_examples, sequence_length as u32],
        inputs.iter().map(|&i| D::from_u32(i)).collect(),
    )
    .unwrap();

    let y_train = one_hot_encode::<T, D>(&targets, vocab.vocab_size, 1)?;

    // --- 3. BUILD MODEL ---
    let mut nn = if !context.restore {
        let mut builder = NeuralNetBuilder::new();

        // Adaptive sizing based on vocabulary size
        // Larger vocabularies need smaller embeddings to fit in 4GB VRAM
        let (head_dim, num_heads) = (4, 4);

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

            let (head_dim, num_heads) = (4, 4);

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
    // --- 4. TRAIN ---
    // Determine epoch offset based on whether we're restoring
    let epoch_offset = if context.restore {
        let restored_epoch = nn.get_current_epoch();
        println!("Resuming training from epoch {}", restored_epoch);
        restored_epoch
    } else {
        0
    };

    let config = TrainingConfig {
        epochs: context.epochs as usize,
        epoch_offset,
        base_lr: D::from_f64(context.learning_rate),
        lr_adjustment: context.lr_adjust,
        weight_normalization: false,
    };

    let mut now = Instant::now();
    if !GLOBAL_CONTEXT.get().unwrap().predict_only {
        nn.fit(
            &x_train,
            &y_train,
            &x_train,
            &y_train,
            config,
            TrainingHook::new(
                context.monitor_interval,
                |e, l, _, _, nn: &mut NeuralNet<T, D>| {
                    println!(" : Loss {:.4}, elapsed {:?}", l, now.elapsed());
                    nn.save_model(&context.weights_path);
                    now = Instant::now();

                    sleep(Duration::from_secs(context.sleep_time));
                },
            ),
        )?;
    }

    // --- 5. GENERATE (with Repetition Penalty) ---
    println!("\nGenerating text...");
    let mut current_window_indices = vec![vocab.stoi["<PAD>"] as u32; sequence_length];
    let mut generated_words = Vec::new();
    let penalty_factor = 1.2; // 1.2 is the "Palash-standard" for balancing diversity

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
        let k = 20; // Only consider the top 20 most likely words
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
