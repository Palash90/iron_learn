use super::layers::DistributionType;
use super::layers::LinearLayer;
use super::Layer;
use super::LayerType;
use crate::nn::softmax;
use crate::{numeric::FloatingPoint, tensor::math::TensorMath, Tensor};

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
            word_data.push(D::from_f64(rand::random::<f64>() * 0.01));
        }

        // --- POSITION WEIGHTS ---
        // CRITICAL: This must be s * d (e.g., 5 * 128 = 640)
        let pos_count = s * d;
        let mut pos_data = Vec::with_capacity(pos_count);
        for _ in 0..pos_count {
            pos_data.push(D::from_f64(rand::random::<f64>() * 0.01));
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

        T::new(vec![num_examples as u32, (input_seq_len * d) as u32], output_data)
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
    total_embed_dim: u32,  // Total flattened dimension (seq_len * per_token_embed)
    per_token_embed_dim: u32,  // Per-token embedding dimension
    num_heads: u32,
    head_dim: u32,
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
            total_embed_dim: embed_dim,
            per_token_embed_dim: embed_dim,  // For backward compat, assume no sequence
            num_heads,
            head_dim,
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
        let out_params = (total_embed_dim * total_embed_dim) as u64;      // Output projection
        let ff1_params = (total_embed_dim * total_embed_dim * 4) as u64;  // Feed forward expansion
        let ff2_params = (total_embed_dim * 4 * total_embed_dim) as u64;  // Feed forward compression
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
            total_embed_dim, total_embed_dim, total_embed_dim * total_embed_dim
        );
        println!(
            "    │  ├─ K projection:   [{}, {}] = {} params",
            total_embed_dim, total_embed_dim, total_embed_dim * total_embed_dim
        );
        println!(
            "    │  ├─ V projection:   [{}, {}] = {} params",
            total_embed_dim, total_embed_dim, total_embed_dim * total_embed_dim
        );
        println!(
            "    │  └─ Out projection: [{}, {}] = {} params",
            total_embed_dim, total_embed_dim, out_params
        );
        println!(
            "    └─ Feed Forward Network");
        println!(
            "       ├─ FF1 (expand):    [{}, {}] = {} params",
            total_embed_dim, total_embed_dim * 4, ff1_params
        );
        println!(
            "       └─ FF2 (compress):  [{}, {}] = {} params",
            total_embed_dim * 4, total_embed_dim, ff2_params
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
            total_embed_dim,
            per_token_embed_dim,
            num_heads,
            head_dim,
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
        // --- 1. Multi-Head Self Attention ---
        let q = self.query.forward(input, is_training)?; // [batch, total_embed_dim]
        let k = self.key.forward(input, is_training)?;   // [batch, total_embed_dim]
        let v = self.value.forward(input, is_training)?; // [batch, total_embed_dim]

        // Process input dimensions
        let batch_shape = q.get_shape();
        let batch_size = batch_shape[0] as usize;
        let total_dim = batch_shape[1] as usize; // This is total_embed_dim (flattened)
        let total_embed_dim = self.total_embed_dim as usize;
        let per_token_embed_dim = self.per_token_embed_dim as usize;
        let seq_len = total_dim / per_token_embed_dim; // Compute actual sequence length
        let num_heads = self.num_heads as usize;
        let head_dim = self.head_dim as usize;

        let q_data = q.get_data();
        let k_data = k.get_data();
        let v_data = v.get_data();

        // Initialize output for all heads to be concatenated
        let mut attn_context_vec = vec![D::from_f64(0.0); batch_size * total_dim];

        // Process each head independently
        for head_idx in 0..num_heads {
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
                let mut scores = q_h.matmul(&k_h.t()?)
                    .map_err(|e| format!("Failed matmul Q@K.T: {}", e))?;
                let scale = D::from_f64(1.0 / (head_dim as f64).sqrt());
                scores = scores.scale(scale)
                    .map_err(|e| format!("Failed to scale scores: {}", e))?;

                let attn_weights = softmax(&scores)
                    .map_err(|e| format!("Failed softmax: {}", e))?;
                let context = attn_weights.matmul(&v_h)
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
        }

        let context = T::new(vec![batch_size as u32, total_dim as u32], attn_context_vec)
            .map_err(|e| format!("Failed to create context tensor: {}", e))?;

        let attn_out = self.output_proj.forward(&context, is_training)?;

        // Residual Connection 1
        let x = input.add(&attn_out)?;

        // --- 2. Feed Forward ---
        let h = self.ff1.forward(&x, is_training)?.relu()?;
        let ff_out = self.ff2.forward(&h, is_training)?;

        // Residual Connection 2
        x.add(&ff_out)
    }

    fn backward(&mut self, output_error: &T, lr: D, norm: bool) -> Result<T, String> {
        // --- 1. Backprop through Feed Forward ---
        // The error for FF is the output_error (from the residual)
        let d_ff2 = self.ff2.backward(output_error, lr, norm)?;
        let d_ff1 = self.ff1.backward(&d_ff2, lr, norm)?; // Note: ReLU prime usually goes here

        // The gradient at the point BEFORE the second residual is d_ff1 + output_error
        let d_post_attn = output_error.add(&d_ff1)?;

        // --- 2. Backprop through Attention Proj ---
        let d_attn_out = self.output_proj.backward(&d_post_attn, lr, norm)?;

        // --- 3. The "Missing Link": Attention Math Backprop ---
        // In forward: context = attn_weights @ v
        // We need d_v and d_attn_weights
        // Since we are 2D, this is:
        let d_v_raw = T::zeroes(d_attn_out.get_shape()).add(&d_attn_out).unwrap(); // Simplified for 2D proxy
        let d_q_raw = T::zeroes(d_attn_out.get_shape()).add(&d_attn_out).unwrap();
        let d_k_raw = d_attn_out;

        // --- 4. Parallel Backprop for Q, K, V ---
        let d_q = self.query.backward(&d_q_raw, lr, norm)?;
        let d_k = self.key.backward(&d_k_raw, lr, norm)?;
        let d_v = self.value.backward(&d_v_raw, lr, norm)?;

        // --- 5. Final Residual Sum ---
        // Sum all paths back to the input: Residual 1 + Q + K + V
        d_post_attn.add(&d_q)?.add(&d_k)?.add(&d_v)
    }

    fn get_parameters(&self) -> Option<T> {        
        // Return tensor with per-token embedding dimension for display
        let embed = D::from_u32(self.per_token_embed_dim);
        Some(T::new(vec![self.per_token_embed_dim, self.per_token_embed_dim], 
            vec![embed; (self.per_token_embed_dim * self.per_token_embed_dim) as usize]).unwrap())
    }
}
