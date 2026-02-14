use super::layers::*;
use crate::nn::loss_functions::LossFunctionType;
use crate::nn::transformer::CombinedEmbedding;
use crate::nn::transformer::TransformerBlock;
use crate::nn::LayerType;
use crate::nn::ModelData;
use crate::numeric::FloatingPoint;
use crate::tensor::math::TensorMath;
use crate::NeuralNet;
use crate::Tensor;
use colored::Colorize;
use crate::examples::contexts::GLOBAL_CONTEXT;

/// Builder type for constructing `NeuralNet` instances via a fluent API.
pub struct NeuralNetBuilder<T, D>
where
    T: TensorMath<D, MathOutput = T>,
    D: FloatingPoint,
{
    layers: Vec<Box<dyn Layer<T, D>>>,
}

impl<T, D> Default for NeuralNetBuilder<T, D>
where
    T: Tensor<D> + TensorMath<D, MathOutput = T> + 'static,
    D: FloatingPoint + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T, D> NeuralNetBuilder<T, D>
where
    T: Tensor<D> + TensorMath<D, MathOutput = T> + 'static,
    D: FloatingPoint + 'static,
{
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }
    /// Add a new fully-connected linear layer to the builder.
    pub fn add_linear(
        &mut self,
        input_size: u32,
        output_size: u32,
        name: &str,
        distribution: &DistributionType,
    ) {
        match LinearLayer::new(input_size, output_size, name, distribution) {
            Ok(layer) => self.layers.push(Box::new(layer)),
            Err(e) => {
                eprintln!("Error adding LinearLayer: {}", e);
            }
        }
    }

    /// Add an activation layer to the builder.
    pub fn add_activation(&mut self, act: LayerType, name: &str) {
        let layer = ActivationLayer::new(name, act);
        self.layers.push(Box::new(layer));
    }

    /// Add an Embedding Layer
    pub fn add_embedding(&mut self, vocab_size: u32, seq_len: u32, embed_dim: u32, name: &str) {
        let layer = CombinedEmbedding::new(vocab_size, seq_len, embed_dim, name);
        self.layers.push(Box::new(layer));
    }

    /// Add a complete multihead transformer setup with embeddings
    /// head_dim: dimension per head (e.g., 16)
    /// num_heads: number of attention heads (e.g., 8)
    /// This will create an embedding with embed_dim = head_dim * num_heads
    pub fn add_multihead_transformer(
        &mut self,
        vocab_size: u32,
        seq_len: u32,
        head_dim: u32,
        num_heads: u32,
        distribution: &DistributionType,
    ) {
        let embed_dim = head_dim * num_heads;
        
        self.add_embedding(vocab_size, seq_len, embed_dim, "embedding");
        self.add_transformer_with_seq(embed_dim, seq_len, num_heads, "transformer", distribution);
    }

    /// Add a Transformer Block for sequence of tokens with per-token multihead attention
    /// per_token_embed_dim: embedding dimension of each token (e.g., 128)
    /// seq_len: number of tokens in sequence (e.g., 5)
    /// num_heads: number of attention heads (e.g., 8)
    pub fn add_transformer_with_seq(
        &mut self,
        per_token_embed_dim: u32,
        seq_len: u32,
        num_heads: u32,
        name: &str,
        distribution: &DistributionType,
    ) {
        let layer = TransformerBlock::multihead_with_seq(name, per_token_embed_dim, seq_len, num_heads, distribution);
        self.layers.push(Box::new(layer));
    }

    /// Add a Transformer Block with single head (default 8 heads)
    pub fn add_transformer_block(
        &mut self,
        embed_dim: u32,
        name: &str,
        distribution: &DistributionType,
    ) {
        let layer = TransformerBlock::with_heads(name, embed_dim, 8, distribution);
        self.layers.push(Box::new(layer));
    }

    /// Add a Transformer Block with specified number of heads for Multi-Head Attention
    pub fn add_transformer_block_with_heads(
        &mut self,
        embed_dim: u32,
        num_heads: u32,
        name: &str,
        distribution: &DistributionType,
    ) {
        let layer = TransformerBlock::with_heads(name, embed_dim, num_heads, distribution);
        self.layers.push(Box::new(layer));
    }

    pub fn build_from_config(
        model: ModelData<D>,
        distribution: &DistributionType,
    ) -> NeuralNet<T, D> {
        let layers = model.layers;
        let loss = model.loss_fn_type;
        let name = model.name;

        // Try to get sequence length from global context
        let seq_len = GLOBAL_CONTEXT
            .get()
            .map(|ctx| ctx.n_gram_size as u32)
            .unwrap_or(1);

        let reconstructed_layers: Vec<Box<dyn Layer<T, D>>> = layers
            .iter()
            .map(|layer_data| match layer_data.layer_type {
                LayerType::Linear => Box::new(
                    LinearLayer::new(
                        layer_data.shape[0],
                        layer_data.shape[1],
                        &layer_data.name,
                        distribution,
                    )
                    .unwrap(),
                ) as Box<dyn Layer<T, D>>,
                LayerType::Embedding => Box::new(CombinedEmbedding::new(
                    layer_data.shape[0],
                    seq_len,
                    layer_data.shape[1],
                    &layer_data.name,
                )) as Box<dyn Layer<T, D>>,

                LayerType::Transformer => Box::new(TransformerBlock::new(
                    &layer_data.name,
                    layer_data.shape[0],
                    distribution,
                )) as Box<dyn Layer<T, D>>,
                _ => Box::new(ActivationLayer::new(
                    layer_data.name.as_str(),
                    layer_data.layer_type.clone(),
                )) as Box<dyn Layer<T, D>>,
            })
            .collect();

        Self::build_network(loss, &name, reconstructed_layers)
    }

    /// Finalize the builder and construct a `NeuralNet`.
    pub fn build(self, loss_fn_type: LossFunctionType, name: &str) -> NeuralNet<T, D> {
        Self::build_network(loss_fn_type, name, self.layers)
    }

    fn build_network(
        loss_fn_type: LossFunctionType,
        name: &str,
        layers: Vec<Box<dyn Layer<T, D>>>,
    ) -> NeuralNet<T, D> {
        println!("Building Network:");
        let mut parameter_count = 0;

        let layer_strings: Vec<String> = layers
            .iter()
            .map(|layer| {
                let name = layer.name();

                match layer.as_ref().get_parameters() {
                    Some(v) => {
                        let shape = v.get_shape();
                        parameter_count += shape.iter().product::<u32>() as u64;
                        format!("{} [{}, {}]", name.bold().cyan(), shape[0], shape[1])
                    }
                    None => {
                        // Format for Activation layers: just the Name
                        format!("{}", name.yellow())
                    }
                }
            })
            .collect();

        let parameter_count = parameter_count as usize;
        let label = if parameter_count >= 1_000_000 {
            format!("{:.1}M", parameter_count as f64 / 1_000_000.0)
        } else if parameter_count >= 1_000 {
            format!("{}k", parameter_count / 1_000)
        } else {
            format!("{}", parameter_count)
        };

        let output = layer_strings.join(" ──▶ ");

        println!("\nModel Architecture ({label}):");
        println!("{}\n", output);

        let network_model = ModelData {
            name: name.to_string(),
            parameter_count: parameter_count as u64,
            layers: Vec::new(), // Layers will be populated during training/saving. This is unsued blank set
            epoch: 0,
            saved_lr: D::zero(),
            loss_fn_type: loss_fn_type.clone(),
            epoch_error: vec![],
            label: label.clone(),
        };

        println!();
        NeuralNet::new(layers, network_model)
    }

    pub fn build_from_model(model: ModelData<D>) -> NeuralNet<T, D> {
        println!("Restoring Model: {}", model.name.bold().green());
        let mut layers: Vec<Box<dyn Layer<T, D>>> = Vec::new();

        // Try to get sequence length from global context for embedding restoration
        let seq_len = GLOBAL_CONTEXT
            .get()
            .map(|ctx| ctx.n_gram_size as u32)
            .unwrap_or(1);

        for layer_data in model.layers {
            match layer_data.layer_type {
                LayerType::Linear => {
                    let weight_tensor = T::new(layer_data.shape, layer_data.weights).unwrap();

                    println!(
                        "Building layer {} with weights {:?}",
                        layer_data.name,
                        weight_tensor.get_shape()
                    );

                    let layer = LinearLayer::from_data(weight_tensor, &layer_data.name);

                    layers.push(Box::new(layer));
                }
                LayerType::Embedding => {
                    println!(
                        "Building layer {} of type Embedding",
                        layer_data.name
                    );

                    // For embedding layers, if we have weights and shape info, load them
                    // Otherwise create with random initialization
                    if !layer_data.shape.is_empty() && layer_data.shape.len() >= 2 {
                        // Create embedding layer with correct seq_len
                        let mut embedding = CombinedEmbedding::new(
                            layer_data.shape[0],
                            seq_len,
                            layer_data.shape[1],
                            &layer_data.name,
                        );
                        
                        // Load saved word embeddings if available
                        if !layer_data.weights.is_empty() {
                            embedding.load_word_embeddings(layer_data.weights, layer_data.shape);
                        }

                        layers.push(Box::new(embedding));
                    } else {
                        // Fallback: create as a regular activation layer (shouldn't normally happen)
                        println!("  Warning: No shape data for embedding layer, creating as activation layer");
                        let layer = ActivationLayer::new(&layer_data.name, layer_data.layer_type);
                        layers.push(Box::new(layer));
                    }
                }
                _ => {
                    println!(
                        "Building layer {} of type {:?}",
                        layer_data.name, layer_data.layer_type
                    );

                    let layer = ActivationLayer::new(&layer_data.name, layer_data.layer_type);
                    layers.push(Box::new(layer));
                }
            }
        }

        println!(
            "Model {} ({}) restored successfully at Epoch {}",
            &model.name, &model.label, &model.epoch
        );

        let model = ModelData {
            name: model.name,
            parameter_count: model.parameter_count,
            layers: vec![], // Layers will be populated during training/saving. This is unsued blank set
            epoch: model.epoch,
            saved_lr: model.saved_lr,
            loss_fn_type: model.loss_fn_type,
            epoch_error: model.epoch_error,
            label: model.label,
        };

        // Construct the final NeuralNet with restored state
        NeuralNet::new(layers, model)
    }
}
