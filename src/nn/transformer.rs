use super::layers::DistributionType;
use super::layers::LinearLayer;
use super::Layer;
use super::LayerType;
use crate::nn::softmax;
use crate::{numeric::FloatingPoint, tensor::math::TensorMath, Tensor};

/// Manual Layer Normalization utility
/// Computes: (x - mean) / sqrt(var + eps) across embedding dimension
/// Due to 2D tensor restriction, operates per-token (per-row) normalization
fn apply_layer_norm<T, D>(input: &T, eps: D) -> Result<T, String>
where
    T: Tensor<D> + TensorMath<D, MathOutput = T> + 'static,
    D: FloatingPoint,
{
    let data = input.get_data();
    let shape = input.get_shape();
    let rows = shape[0] as usize;
    let cols = shape[1] as usize;

    let mut normalized = Vec::with_capacity(rows * cols);

    // Normalize each row (token) independently across embedding dimension
    for r in 0..rows {
        let start = r * cols;
        let end = start + cols;
        let row_data = &data[start..end];

        // Compute mean
        let sum: D = row_data.iter().fold(D::zero(), |acc, &x| acc + x);
        let mean = sum / D::from_u32(cols as u32);

        // Compute variance
        let mut variance = D::zero();
        for &val in row_data {
            let diff = val - mean;
            variance = variance + (diff * diff);
        }
        variance = variance / D::from_u32(cols as u32);

        // Normalize
        let std_dev = (variance + eps).sqrt();
        for &val in row_data {
            let normalized_val = (val - mean) / std_dev;
            normalized.push(normalized_val);
        }
    }

    T::new(shape.clone(), normalized)
}

/// Apply causal mask to attention scores (2D matrix [seq_len, seq_len])
/// Masks out future positions by setting them to -infinity for language modeling
fn apply_causal_mask<T, D>(scores: &T) -> Result<T, String>
where
    T: Tensor<D> + TensorMath<D, MathOutput = T> + 'static,
    D: FloatingPoint,
{
    let mut scores_data = scores.get_data();
    let shape = scores.get_shape();
    let seq_len = shape[0] as usize;

    // Set upper triangle (future positions) to -infinity
    // This ensures softmax will make them nearly zero
    let neg_inf = D::from_f64(f64::NEG_INFINITY);
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            scores_data[i * seq_len + j] = neg_inf;
        }
    }

    T::new(shape.clone(), scores_data)
}

pub struct CombinedEmbedding<T, D> {
    name: String,
    word_weights: T, // [vocab_size, embed_dim]
    pos_weights: T,  // [seq_len, embed_dim]
    vocab_size: u32,
    seq_len: u32,
    embed_dim: u32,
    last_input: Option<T>,
    _marker: std::marker::PhantomData<D>,
}

impl<T, D> CombinedEmbedding<T, D>
where
    T: Tensor<D> + TensorMath<D, MathOutput = T> + 'static,
    D: FloatingPoint,
{
    pub fn new(vocab_size: u32, seq_len: u32, embed_dim: u32, name: &str) -> Self {
        let d = embed_dim as usize;
        let s = seq_len as usize;

        // --- WORD WEIGHTS ---
        let word_count = vocab_size as usize * d;
        let mut word_data = Vec::with_capacity(word_count);
        for _ in 0..word_count {
            let val = rand::random::<f64>() - 0.5; // Center at 0
            word_data.push(D::from_f64(val));
        }

        // --- POSITION WEIGHTS ---
        // CRITICAL: This must be s * d (e.g., 5 * 128 = 640)
        let pos_count = s * d;
        let mut pos_data = Vec::with_capacity(pos_count);
        for _ in 0..pos_count {
            let val = rand::random::<f64>() - 0.5; // Center at 0
            pos_data.push(D::from_f64(val));
        }

        // Print embedding architecture details
        let word_params = (vocab_size * embed_dim) as u64;
        let pos_params = (seq_len * embed_dim) as u64;
        let total_params = word_params + pos_params;
        println!(
            "  Embedding '{}': vocab_size={}, seq_len={}, embed_dim={}, total_params={}",
            name, vocab_size, seq_len, embed_dim, total_params
        );
        println!(
            "    ├─ Word Embeddings:     [vocab={}, dim={}] = {} params",
            vocab_size, embed_dim, word_params
        );
        println!(
            "    └─ Position Embeddings: [seq={}, dim={}] = {} params",
            seq_len, embed_dim, pos_params
        );

        Self {
            name: name.to_string(),
            word_weights: T::new(vec![vocab_size, embed_dim], word_data).unwrap(),
            pos_weights: T::new(vec![seq_len, embed_dim], pos_data).unwrap(), // Shape [5, 128]
            vocab_size,
            seq_len,
            embed_dim,
            last_input: None,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn load_word_embeddings(&mut self, weights: Vec<D>, shape: Vec<u32>) {
        // Load saved word embeddings
        if shape.len() == 2 {
            self.word_weights = T::new(shape, weights).unwrap();
        }
    }
}

impl<T, D> Layer<T, D> for CombinedEmbedding<T, D>
where
    T: Tensor<D> + TensorMath<D, MathOutput = T> + 'static,
    D: FloatingPoint,
{
    fn forward(&mut self, input: &T, is_training: bool) -> Result<T, String> {
        if is_training {
            self.last_input = Some(T::zeroes(input.get_shape()).add(input).unwrap());
        }

        let indices = input.get_data();
        let word_w = self.word_weights.get_data(); // Length: vocab * 128

        let d = self.embed_dim as usize;
        let shape = input.get_shape();
        let num_examples = shape[0] as usize;
        let input_seq_len = shape[1] as usize;

        // If the incoming sequence length differs from initialized seq_len,
        // expand pos_weights to match
        if input_seq_len != self.seq_len as usize {
            // Reinitialize position weights if needed
            let s = input_seq_len;
            let pos_count = s * d;
            let mut pos_data = Vec::with_capacity(pos_count);

            // Reuse existing position embeddings if they're smaller, pad with new random ones
            let existing_pos = self.pos_weights.get_data();
            let existing_seq_len = self.seq_len as usize;

            for pos_idx in 0..s {
                for embed_idx in 0..d {
                    let value = if pos_idx < existing_seq_len {
                        // Reuse existing embedding
                        existing_pos[pos_idx * d + embed_idx]
                    } else {
                        // Generate new random embedding
                        D::from_f64(rand::random::<f64>() * 0.01)
                    };
                    pos_data.push(value);
                }
            }

            self.pos_weights = T::new(vec![input_seq_len as u32, self.embed_dim], pos_data)?;
            self.seq_len = input_seq_len as u32;
        }

        let pos_w = self.pos_weights.get_data(); // Length: input_seq_len * 128

        let mut output_data = Vec::with_capacity(num_examples * input_seq_len * d);

        for b in 0..num_examples {
            for s in 0..input_seq_len {
                // Word Vector Lookup
                let token_idx = indices[b * input_seq_len + s].f32() as usize;
                let w_start = token_idx * d;
                let word_slice = &word_w[w_start..w_start + d];

                // Position Vector Lookup
                let p_start = s * d;
                let pos_slice = &pos_w[p_start..p_start + d];

                for i in 0..d {
                    output_data.push(word_slice[i] + pos_slice[i]);
                }
            }
        }

        T::new(
            vec![num_examples as u32, (input_seq_len * d) as u32],
            output_data,
        )
    }

    fn backward(&mut self, output_error: &T, lr: D, _norm: bool) -> Result<T, String> {
        let input = self.last_input.as_ref().ok_or("No input for backward")?;
        let indices = input.get_data();
        let error_data = output_error.get_data();

        let num_examples = input.get_shape()[0] as usize;
        let seq_len = input.get_shape()[1] as usize;
        let embed_dim = self.embed_dim as usize;

        // Create mutable copies of weights to update
        let mut word_weight_data = self.word_weights.get_data().to_vec();
        let mut pos_weight_data = self.pos_weights.get_data().to_vec();

        for b in 0..num_examples {
            for s in 0..seq_len {
                // 1. Identify which word and which position we are looking at
                let token_idx = indices[b * seq_len + s].f32() as usize;
                let pos_idx = s; // The position is simply the sequence index

                // 2. Locate the error vector for this specific token in the batch
                // output_error shape is [Batch, SeqLen * EmbedDim]
                let error_offset = (b * seq_len * embed_dim) + (s * embed_dim);
                let current_error = &error_data[error_offset..error_offset + embed_dim];

                // 3. Update Word Weights
                let w_start = token_idx * embed_dim;
                for i in 0..embed_dim {
                    let grad = current_error[i];
                    word_weight_data[w_start + i] = word_weight_data[w_start + i] - (grad * lr);
                }

                // 4. Update Position Weights
                let p_start = pos_idx * embed_dim;
                for i in 0..embed_dim {
                    let grad = current_error[i];
                    pos_weight_data[p_start + i] = pos_weight_data[p_start + i] - (grad * lr);
                }
            }
        }

        // Save updated weights back to tensors
        self.word_weights = T::new(vec![self.vocab_size, self.embed_dim], word_weight_data)?;
        self.pos_weights = T::new(vec![self.seq_len, self.embed_dim], pos_weight_data)?;

        // Return zeroed tensor matching input shape (indices don't receive gradients)
        Ok(T::zeroes(input.get_shape()))
    }

    fn get_parameters(&self) -> Option<T> {
        // Return word embeddings for serialization
        // Position embeddings will be regenerated on restore
        Some(
            T::zeroes(self.word_weights.get_shape())
                .add(&self.word_weights)
                .unwrap(),
        )
    }

    fn name(&self) -> &str {
        &self.name
    }
    fn layer_type(&self) -> &LayerType {
        &LayerType::Embedding
    }
}

pub struct TransformerCache<T> {
    q: T,
    k: T,
    v: T,
    attn_weights: Vec<Vec<T>>, // Store weights per head
}

pub struct TransformerBlock<T, D>
where
    T: Tensor<D> + TensorMath<D, MathOutput = T> + 'static,
    D: FloatingPoint,
{
    name: String,
    layer_type: LayerType,
    // Multi-head Attention Projections
    query: LinearLayer<T, D>,
    key: LinearLayer<T, D>,
    value: LinearLayer<T, D>,
    output_proj: LinearLayer<T, D>,
    // Feed Forward Network
    ff1: LinearLayer<T, D>,
    ff2: LinearLayer<T, D>,
    per_token_embed_dim: u32, // Per-token embedding dimension
    num_heads: u32,
    head_dim: u32,
    // Layer Normalization epsilon value
    ln_eps: D,
    // Enable causal masking for language modeling
    use_causal_mask: bool,

    cache: Option<TransformerCache<T>>,
}

impl<T, D> TransformerBlock<T, D>
where
    T: Tensor<D> + TensorMath<D, MathOutput = T> + 'static,
    D: FloatingPoint,
{
    pub fn new(name: &str, embed_dim: u32, dist: &DistributionType) -> Self {
        Self::with_heads(name, embed_dim, 8, dist)
    }

    pub fn with_heads(name: &str, embed_dim: u32, num_heads: u32, dist: &DistributionType) -> Self {
        assert!(
            embed_dim.is_multiple_of(num_heads),
            "embed_dim {} must be divisible by num_heads {}",
            embed_dim,
            num_heads
        );
        let head_dim = embed_dim / num_heads;
        Self {
            name: name.to_string(),
            layer_type: LayerType::Transformer,
            // Project input using total dimension (will be reshape-aware in forward)
            query: LinearLayer::new(embed_dim, embed_dim, "q", dist).unwrap(),
            key: LinearLayer::new(embed_dim, embed_dim, "k", dist).unwrap(),
            value: LinearLayer::new(embed_dim, embed_dim, "v", dist).unwrap(),
            output_proj: LinearLayer::new(embed_dim, embed_dim, "out", dist).unwrap(),
            ff1: LinearLayer::new(embed_dim, embed_dim * 4, "ff1", dist).unwrap(),
            ff2: LinearLayer::new(embed_dim * 4, embed_dim, "ff2", dist).unwrap(),
            per_token_embed_dim: embed_dim, // For backward compat, assume no sequence
            num_heads,
            head_dim,
            ln_eps: D::from_f64(1e-6),
            use_causal_mask: false,
            cache: None, // Cache will be used in autoregressive decoding, not implemented here
        }
    }

    /// Create transformer for multihead attention on sequence of tokens
    /// per_token_embed_dim: embedding dimension of each token (e.g., 128)
    /// seq_len: number of tokens in sequence (e.g., 5)
    /// num_heads: number of attention heads (e.g., 8)
    pub fn multihead_with_seq(
        name: &str,
        per_token_embed_dim: u32,
        seq_len: u32,
        num_heads: u32,
        dist: &DistributionType,
    ) -> Self {
        assert!(
            per_token_embed_dim.is_multiple_of(num_heads),
            "per_token_embed_dim {} must be divisible by num_heads {}",
            per_token_embed_dim,
            num_heads
        );
        let total_embed_dim = per_token_embed_dim * seq_len;
        let head_dim = per_token_embed_dim / num_heads;

        // Calculate parameter counts for display
        let qkv_params = (total_embed_dim * total_embed_dim) as u64 * 3; // Q, K, V projections
        let out_params = (total_embed_dim * total_embed_dim) as u64; // Output projection
        let ff1_params = (total_embed_dim * total_embed_dim * 4) as u64; // Feed forward expansion
        let ff2_params = (total_embed_dim * 4 * total_embed_dim) as u64; // Feed forward compression
        let total_params = qkv_params + out_params + ff1_params + ff2_params;

        println!(
            "  Transformer '{}': num_heads={}, head_dim={}, seq_len={}, per_token_dim={}, total_dim={}, total_params={}",
            name, num_heads, head_dim, seq_len, per_token_embed_dim, total_embed_dim, total_params
        );
        println!(
            "    ├─ Multi-Head Attention ({} heads × {} dims each)",
            num_heads, head_dim
        );
        println!(
            "    │  ├─ Q projection:   [{}, {}] = {} params",
            total_embed_dim,
            total_embed_dim,
            total_embed_dim * total_embed_dim
        );
        println!(
            "    │  ├─ K projection:   [{}, {}] = {} params",
            total_embed_dim,
            total_embed_dim,
            total_embed_dim * total_embed_dim
        );
        println!(
            "    │  ├─ V projection:   [{}, {}] = {} params",
            total_embed_dim,
            total_embed_dim,
            total_embed_dim * total_embed_dim
        );
        println!(
            "    │  └─ Out projection: [{}, {}] = {} params",
            total_embed_dim, total_embed_dim, out_params
        );
        println!("    └─ Feed Forward Network");
        println!(
            "       ├─ FF1 (expand):    [{}, {}] = {} params",
            total_embed_dim,
            total_embed_dim * 4,
            ff1_params
        );
        println!(
            "       └─ FF2 (compress):  [{}, {}] = {} params",
            total_embed_dim * 4,
            total_embed_dim,
            ff2_params
        );

        Self {
            name: name.to_string(),
            layer_type: LayerType::Transformer,
            // Projections work on flattened total dimension
            query: LinearLayer::new(total_embed_dim, total_embed_dim, "q", dist).unwrap(),
            key: LinearLayer::new(total_embed_dim, total_embed_dim, "k", dist).unwrap(),
            value: LinearLayer::new(total_embed_dim, total_embed_dim, "v", dist).unwrap(),
            output_proj: LinearLayer::new(total_embed_dim, total_embed_dim, "out", dist).unwrap(),
            ff1: LinearLayer::new(total_embed_dim, total_embed_dim * 4, "ff1", dist).unwrap(),
            ff2: LinearLayer::new(total_embed_dim * 4, total_embed_dim, "ff2", dist).unwrap(),
            per_token_embed_dim,
            num_heads,
            head_dim,
            ln_eps: D::from_f64(1e-6),
            use_causal_mask: true,  // Enable causal masking for language models
            cache: None,
        }
    }

    fn extract_head_slice(
        &self,
        tensor: &T,
        batch_idx: usize,
        head_idx: usize,
        seq_len: usize,
    ) -> Result<T, String> {
        let data = tensor.get_data();
        let total_dim = seq_len * self.per_token_embed_dim as usize;
        let per_token_dim = self.per_token_embed_dim as usize;
        let h_dim = self.head_dim as usize;

        let mut head_data = Vec::with_capacity(seq_len * h_dim);

        for s in 0..seq_len {
            // Offset logic: Start of batch + Start of token + Start of head
            let start = (batch_idx * total_dim) + (s * per_token_dim) + (head_idx * h_dim);
            head_data.extend_from_slice(&data[start..start + h_dim]);
        }

        T::new(vec![seq_len as u32, h_dim as u32], head_data)
    }

    fn map_head_back_to_buffer(
        &self,
        buffer: &mut Vec<D>,
        head_grad: &T,
        batch_idx: usize,
        head_idx: usize,
        seq_len: usize,
        total_dim: usize,
    ) {
        let grad_data = head_grad.get_data();
        let per_token_dim = self.per_token_embed_dim as usize;
        let h_dim = self.head_dim as usize;

        for s in 0..seq_len {
            let buffer_offset = (batch_idx * total_dim) + (s * per_token_dim) + (head_idx * h_dim);
            let grad_offset = s * h_dim;

            for i in 0..h_dim {
                buffer[buffer_offset + i] = grad_data[grad_offset + i];
            }
        }
    }
}

impl<T, D> Layer<T, D> for TransformerBlock<T, D>
where
    T: Tensor<D> + TensorMath<D, MathOutput = T> + 'static,
    D: FloatingPoint,
{
    fn name(&self) -> &str {
        &self.name
    }
    fn layer_type(&self) -> &LayerType {
        &self.layer_type
    }

    fn forward(&mut self, input: &T, is_training: bool) -> Result<T, String> {
        // --- 0. Layer Normalization before Attention (Pre-norm architecture) ---
        let normalized_input = apply_layer_norm(input, self.ln_eps)
            .map_err(|e| format!("Failed layer norm before attention: {}", e))?;

        // --- 1. Multi-Head Self Attention ---
        let q = self.query.forward(&normalized_input, is_training)?; // [batch, total_embed_dim]
        let k = self.key.forward(&normalized_input, is_training)?; // [batch, total_embed_dim]
        let v = self.value.forward(&normalized_input, is_training)?; // [batch, total_embed_dim]

        // Process input dimensions
        let batch_shape = q.get_shape();
        let batch_size = batch_shape[0] as usize;
        let total_dim = batch_shape[1] as usize; // This is total_embed_dim (flattened)
        let per_token_embed_dim = self.per_token_embed_dim as usize;
        let seq_len = total_dim / per_token_embed_dim; // Compute actual sequence length
        let num_heads = self.num_heads as usize;
        let head_dim = self.head_dim as usize;

        let q_data = q.get_data();
        let k_data = k.get_data();
        let v_data = v.get_data();

        // Initialize output for all heads to be concatenated
        let mut attn_context_vec = vec![D::from_f64(0.0); batch_size * total_dim];
        let mut all_head_weights = Vec::with_capacity(num_heads);

        // Process each head independently
        for head_idx in 0..num_heads {
            let mut head_weights_batch = Vec::with_capacity(batch_size);
            for batch_idx in 0..batch_size {
                // Extract Q, K, V for this head
                let mut q_head = Vec::with_capacity(seq_len * head_dim);
                let mut k_head = Vec::with_capacity(seq_len * head_dim);
                let mut v_head = Vec::with_capacity(seq_len * head_dim);

                for seq_idx in 0..seq_len {
                    for d in 0..head_dim {
                        let idx = batch_idx * seq_len * per_token_embed_dim
                            + seq_idx * per_token_embed_dim
                            + head_idx * head_dim
                            + d;
                        q_head.push(q_data[idx]);
                        k_head.push(k_data[idx]);
                        v_head.push(v_data[idx]);
                    }
                }

                // Create tensors for this head: [seq_len, head_dim]
                let q_h = T::new(vec![seq_len as u32, head_dim as u32], q_head)
                    .map_err(|e| format!("Failed to create Q head: {}", e))?;
                let k_h = T::new(vec![seq_len as u32, head_dim as u32], k_head)
                    .map_err(|e| format!("Failed to create K head: {}", e))?;
                let v_h = T::new(vec![seq_len as u32, head_dim as u32], v_head)
                    .map_err(|e| format!("Failed to create V head: {}", e))?;

                // Attention Score = Softmax( (Q @ K.T) / sqrt(head_dim) )
                let mut scores = q_h
                    .matmul(&k_h.t()?)
                    .map_err(|e| format!("Failed matmul Q@K.T: {}", e))?;
                let scale = D::from_f64(1.0 / (head_dim as f64).sqrt());
                scores = scores
                    .scale(scale)
                    .map_err(|e| format!("Failed to scale scores: {}", e))?;

                // CRITICAL: Apply causal masking for language models
                // This prevents attending to future positions during self-attention
                if self.use_causal_mask {
                    scores = apply_causal_mask(&scores)
                        .map_err(|e| format!("Failed to apply causal mask: {}", e))?;
                }

                let attn_weights =
                    softmax(&scores).map_err(|e| format!("Failed softmax: {}", e))?;
                head_weights_batch.push(
                    T::zeroes(attn_weights.get_shape())
                        .add(&attn_weights)
                        .unwrap(),
                );

                let context = attn_weights
                    .matmul(&v_h)
                    .map_err(|e| format!("Failed attention context: {}", e))?;

                // Write attention output for this head directly to final position
                let context_data = context.get_data();
                for seq_idx in 0..seq_len {
                    for d in 0..head_dim {
                        // Position in final output: batch, seq, head concatenated
                        let out_pos = batch_idx * seq_len * per_token_embed_dim
                            + seq_idx * per_token_embed_dim
                            + head_idx * head_dim
                            + d;
                        attn_context_vec[out_pos] = context_data[seq_idx * head_dim + d];
                    }
                }
            }
            all_head_weights.push(head_weights_batch);
        }

        let context = T::new(vec![batch_size as u32, total_dim as u32], attn_context_vec)
            .map_err(|e| format!("Failed to create context tensor: {}", e))?;

        if is_training {
            // CRITICAL: Save for backward pass
            self.cache = Some(TransformerCache {
                q: T::zeroes(q.get_shape()).add(&q).unwrap(),
                k: T::zeroes(k.get_shape()).add(&k).unwrap(),
                v: T::zeroes(v.get_shape()).add(&v).unwrap(),
                attn_weights: all_head_weights,
            });
        }

        let attn_out = self.output_proj.forward(&context, is_training)?;

        // Residual Connection 1
        let x = input.add(&attn_out)?;

        // --- 2. Feed Forward with Layer Normalization (Pre-norm architecture) ---
        let normalized_x = apply_layer_norm(&x, self.ln_eps)
            .map_err(|e| format!("Failed layer norm before FFN: {}", e))?;
        let h = self.ff1.forward(&normalized_x, is_training)?.relu()?;
        let ff_out = self.ff2.forward(&h, is_training)?;

        // Residual Connection 2
        x.add(&ff_out)
    }

    fn backward(&mut self, output_error: &T, lr: D, norm: bool) -> Result<T, String> {
        // CRITICAL FIX: Proper residual gradient accumulation
        // For residual z = x + f(x), gradient is: dz/dx = 1 (from residual) + df/dx (from function)
        // Both branches must contribute to the final gradient
        
        // 1. Backward through FF branch
        let d_ff2 = self.ff2.backward(output_error, lr, norm)?;
        let d_ff1 = self.ff1.backward(&d_ff2, lr, norm)?;
        
        // 2. Gradient accumulation: output_error flows through both
        //    - Directly through residual connection: output_error
        //    - Through FF layers: d_ff1
        let d_x_after_attn = output_error.add(&d_ff1)?;
        
        // 3. Backward through attention output projection residual
        let d_context_full = self.output_proj.backward(&d_x_after_attn, lr, norm)?;

        // 2. Setup Dimensions & Retrieve Cache
        let cache = self
            .cache
            .as_ref()
            .ok_or("No cache found for backward pass")?;
        let batch_size = d_context_full.get_shape()[0] as usize;
        let total_dim = d_context_full.get_shape()[1] as usize;
        let seq_len = total_dim / (self.per_token_embed_dim as usize);
        let h_dim = self.head_dim as usize;
        let n_heads = self.num_heads as usize;
        let scale = D::from_f64(1.0 / (h_dim as f64).sqrt());

        let mut dq_final = vec![D::zero(); batch_size * total_dim];
        let mut dk_final = vec![D::zero(); batch_size * total_dim];
        let mut dv_final = vec![D::zero(); batch_size * total_dim];

        // 3. Manual Head-wise Loop logic
        for h in 0..n_heads {
            for b in 0..batch_size {
                // --- Extract d_head (Incoming Gradient) ---
                let mut d_head_data = Vec::with_capacity(seq_len * h_dim);
                for s in 0..seq_len {
                    let start = b * total_dim + s * (self.per_token_embed_dim as usize) + h * h_dim;
                    d_head_data.extend_from_slice(&d_context_full.get_data()[start..start + h_dim]);
                }
                let d_head = T::new(vec![seq_len as u32, h_dim as u32], d_head_data)?;

                // --- Retrieve Forward States for this Head/Batch ---
                // Note: You'll need helper methods to slice your cached Q, K, V similar to forward
                let q_h = self.extract_head_slice(&cache.q, b, h, seq_len)?;
                let k_h = self.extract_head_slice(&cache.k, b, h, seq_len)?;
                let v_h = self.extract_head_slice(&cache.v, b, h, seq_len)?;
                let weights = &cache.attn_weights[h][b];

                // --- THE ATTENTION GRADIENT MATH ---

                // 1. Gradient w.r.t V: Weights^T * d_head
                let d_v_h = weights.t()?.matmul(&d_head)?;

                // 2. Gradient w.r.t Softmax Scores
                let d_weights = d_head.matmul(&v_h.t()?)?;

                // 3. Proper Softmax Backward (Manual Row-wise)
                let s_data = weights.get_data();
                let ds_data = d_weights.get_data();
                let mut d_scores_data = Vec::with_capacity(seq_len * seq_len);

                for r in 0..seq_len {
                    // Calculate dot product for this row: sum(S_i * dS_i)
                    let mut row_dot_product = D::zero();
                    for c in 0..seq_len {
                        let idx = r * seq_len + c;
                        row_dot_product = row_dot_product + (s_data[idx] * ds_data[idx]);
                    }

                    // Calculate d_scores for this row: S_i * (dS_i - row_dot_product)
                    for c in 0..seq_len {
                        let idx = r * seq_len + c;
                        d_scores_data.push(s_data[idx] * (ds_data[idx] - row_dot_product));
                    }
                }
                let d_scores = T::new(vec![seq_len as u32, seq_len as u32], d_scores_data)?;

                // 4. Gradient w.r.t Q and K
                let d_q_h = d_scores.matmul(&k_h)?.scale(scale)?;
                let d_k_h = d_scores.t()?.matmul(&q_h)?.scale(scale)?;

                // --- Re-insert gradients into the flattened buffers ---
                self.map_head_back_to_buffer(&mut dq_final, &d_q_h, b, h, seq_len, total_dim);
                self.map_head_back_to_buffer(&mut dk_final, &d_k_h, b, h, seq_len, total_dim);
                self.map_head_back_to_buffer(&mut dv_final, &d_v_h, b, h, seq_len, total_dim);
            }
        }

        // 4. Update Q, K, V Projections
        let d_q = self.query.backward(
            &T::new(vec![batch_size as u32, total_dim as u32], dq_final)?,
            lr,
            norm,
        )?;
        let d_k = self.key.backward(
            &T::new(vec![batch_size as u32, total_dim as u32], dk_final)?,
            lr,
            norm,
        )?;
        let d_v = self.value.backward(
            &T::new(vec![batch_size as u32, total_dim as u32], dv_final)?,
            lr,
            norm,
        )?;

        // Combine for final input error from all branches
        d_x_after_attn.add(&d_q)?.add(&d_k)?.add(&d_v)
    }

    fn get_parameters(&self) -> Option<T> {
        // Return tensor with per-token embedding dimension for display
        let embed = D::from_u32(self.per_token_embed_dim);
        Some(
            T::new(
                vec![self.per_token_embed_dim, self.per_token_embed_dim],
                vec![embed; (self.per_token_embed_dim * self.per_token_embed_dim) as usize],
            )
            .unwrap(),
        )
    }
}
