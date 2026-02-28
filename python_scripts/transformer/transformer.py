import cupy as np
from simple_layers import Linear, ReLU, Softmax

try:
    np.cuda.Device(0).use()
    print("GPU found and will be used for training.")
except Exception as e:
    print(f"GPU not found or error: {e}")

class Embedding:
    def __init__(self, vocab_size, d_model):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.weight = np.random.randn(vocab_size, d_model) * np.sqrt(2.0 / vocab_size)

    def forward(self, x):
        assert x.ndim == 1, "Input should be a 1D array of token indices"
        self.inputs = x
        return self.weight[x]
    
    def backward(self, grad_output, learning_rate):
        np.add.at(self.weight, self.inputs, -learning_rate * grad_output)
        return None

class PositionalEncoding:
    def __init__(self, embedding_dimension, sequence_length):
        self.embedding_dimension = embedding_dimension
        self.sequence_length = sequence_length
        self.weight = np.random.randn(self.sequence_length, self.embedding_dimension) * 0.02
    
    def forward(self, x):
        assert x.ndim == 2 and x.shape[1] == self.embedding_dimension, "Input should be of shape (sequence_length, embedding_dimension)"
        self.inputs = x
        return x + self.weight[:x.shape[0], :]
    
    def backward(self, grad_output, learning_rate):
        # Correctly update positional encoding weights
        seq_len = grad_output.shape[0]
        self.weight[:seq_len] -= learning_rate * grad_output
        return grad_output
class Dropout:
    def __init__(self, p=0.1):
        self.p = p
        self.mask = None

    def forward(self, x, training=True):
        if not training or self.p == 0:
            return x
        # Create a mask of 1s and 0s. 
        # (1-p) is the probability of keeping the neuron active.
        self.mask = (np.random.rand(*x.shape) > self.p)
        # Scale the output by 1/(1-p) so the expected value remains the same
        return (x * self.mask) / (1.0 - self.p)

    def backward(self, grad_output):
        if self.mask is None:
            return grad_output
        return (grad_output * self.mask) / (1.0 - self.p)
    
class Attention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = Linear(d_model, d_model)
        self.W_k = Linear(d_model, d_model)
        self.W_v = Linear(d_model, d_model)
        self.W_o = Linear(d_model, d_model)
        self.softmax_layers = [Softmax() for _ in range(num_heads)]

        self.q_full = None
        self.k_full = None
        self.v_full = None
        self.attention_probs = {}
        self.inputs = None
        self.causal_mask = None
        
    
    def forward(self, x):
        assert x.ndim == 2 and x.shape[1] == self.d_model, "Input should be of shape (sequence_length, d_model)"
        self.inputs = x

        self.q_full = self.W_q.forward(x)
        self.k_full = self.W_k.forward(x)
        self.v_full = self.W_v.forward(x)

        self.causal_mask = np.triu(np.ones((x.shape[0], x.shape[0])) * -1e9, k=1)

        head_outputs = []

        for i in range(self.num_heads):
            start, end = i * self.d_k, (i + 1) * self.d_k
            q = self.q_full[:, start:end]
            k = self.k_full[:, start:end]
            v = self.v_full[:, start:end]

            attention_scores = np.dot(q, k.T) / np.sqrt(self.d_k)
            attention_scores += self.causal_mask
            self.attention_probs[i] = self.softmax_layers[i].forward(attention_scores)

            head_output = np.dot(self.attention_probs[i], v)

            head_outputs.append(head_output)

        attention_output = np.concatenate(head_outputs, axis=-1)
        return self.W_o.forward(attention_output)
    
    def backward(self, grad_output, learning_rate):
        grad_attention_output = self.W_o.backward(grad_output, learning_rate)

        grad_q_full = np.zeros_like(self.q_full)
        grad_k_full = np.zeros_like(self.k_full)
        grad_v_full = np.zeros_like(self.v_full)

        for i in range(self.num_heads):
            start, end = i * self.d_k, (i + 1) * self.d_k
            grad_head_output = grad_attention_output[:, start:end]

            q, k, v = self.q_full[:, start:end], self.k_full[:, start:end], self.v_full[:, start:end]
            probs = self.attention_probs[i]

            # Gradient w.r.t. values
            grad_v_full[:, start:end] = np.dot(probs.T, grad_head_output)
            # Gradient w.r.t. attention probabilities
            grad_probs = np.dot(grad_head_output, v.T)

            # Gradient w.r.t. attention scores (before softmax, after mask)
            grad_scores = self.softmax_layers[i].backward(grad_probs) / np.sqrt(self.d_k)
            # Apply mask: gradients at masked positions should be zero
            grad_scores = grad_scores * (self.causal_mask == 0)

            # Gradient w.r.t. queries and keys
            grad_q_full[:, start:end] = np.dot(grad_scores, k)
            grad_k_full[:, start:end] = np.dot(grad_scores.T, q)

        grad_input_q = self.W_q.backward(grad_q_full, learning_rate)
        grad_input_k = self.W_k.backward(grad_k_full, learning_rate)
        grad_input_v = self.W_v.backward(grad_v_full, learning_rate)

        return grad_input_q + grad_input_k + grad_input_v
    
class Attention_ND:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = Linear(d_model, d_model)
        self.W_k = Linear(d_model, d_model)
        self.W_v = Linear(d_model, d_model)
        self.W_o = Linear(d_model, d_model)
        self.softmax = Softmax() # One softmax for the whole tensor

    def forward(self, x):
        self.inputs = x
        seq_len = x.shape[0]

        # 1. Linear projections
        q = self.W_q.forward(x) # (seq_len, d_model)
        k = self.W_k.forward(x)
        v = self.W_v.forward(x)

        # 2. Reshape for Multi-Head: (seq_len, heads, d_k) 
        # then transpose to (heads, seq_len, d_k)
        self.q = q.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)
        self.k = k.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)
        self.v = v.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)

        # 3. Scaled Dot-Product Attention (Parallel over heads)
        # scores: (heads, seq_len, seq_len)
        scores = np.matmul(self.q, self.k.transpose(0, 2, 1)) / np.sqrt(self.d_k)
        
        # Causal mask
        mask = np.triu(np.ones((seq_len, seq_len)) * -1e9, k=1)
        scores += mask

        self.probs = self.softmax.forward(scores)
        
        # 4. Concatenate heads
        # out: (heads, seq_len, d_k) -> (seq_len, heads, d_k) -> (seq_len, d_model)
        out = np.matmul(self.probs, self.v)
        out = out.transpose(1, 0, 2).reshape(seq_len, self.d_model)
        
        return self.W_o.forward(out)

    def backward(self, grad_output, learning_rate):
        # Backward through W_o
        grad_out = self.W_o.backward(grad_output, learning_rate)
        seq_len = grad_out.shape[0]

        # Reshape back to heads
        grad_out = grad_out.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)

        # Gradient w.r.t V
        grad_v = np.matmul(self.probs.transpose(0, 2, 1), grad_out)
        
        # Gradient w.r.t Probs
        grad_probs = np.matmul(grad_out, self.v.transpose(0, 2, 1))
        
        # Gradient w.r.t Scores
        grad_scores = self.softmax.backward(grad_probs) / np.sqrt(self.d_k)
        
        # Gradient w.r.t Q and K
        grad_q = np.matmul(grad_scores, self.k)
        grad_k = np.matmul(grad_scores.transpose(0, 2, 1), self.q)

        # Reshape and transpose back to (seq_len, d_model)
        grad_q = grad_q.transpose(1, 0, 2).reshape(seq_len, self.d_model)
        grad_k = grad_k.transpose(1, 0, 2).reshape(seq_len, self.d_model)
        grad_v = grad_v.transpose(1, 0, 2).reshape(seq_len, self.d_model)

        # Backward through initial Linear layers
        return (self.W_q.backward(grad_q, learning_rate) + 
                self.W_k.backward(grad_k, learning_rate) + 
                self.W_v.backward(grad_v, learning_rate))

class FeedForward:
    def __init__(self, d_model, d_ff):
        self.linear1 = Linear(d_model, d_ff)
        self.relu = ReLU()
        self.linear2 = Linear(d_ff, d_model)

        
    def forward(self, x):
        x = self.linear1.forward(x)
        x = self.relu.forward(x)
        x = self.linear2.forward(x)
        return x
    
    def backward(self, grad_output, learning_rate):
        grad_output = self.linear2.backward(grad_output, learning_rate)
        grad_output = self.relu.backward(grad_output)
        grad_output = self.linear1.backward(grad_output, learning_rate)
        return grad_output

class ResidualConnection:
    def __init__(self):
        pass

    def forward(self, x, sublayer_output):
        return x + sublayer_output
    
    def backward(self, grad_output):
        return grad_output, grad_output

class LayerNorm:
    def __init__(self, d_model, eps=1e-6):
        self.d_model = d_model
        self.eps = eps
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)

    def forward(self, x):
        self.inputs = x
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)
        self.x_normalized = (x - self.mean) / np.sqrt(self.var + self.eps)
        return self.gamma * self.x_normalized + self.beta
    
    def backward(self, grad_output, learning_rate):
        # Gradient w.r.t. scale (gamma) and shift (beta) parameters
        grad_gamma = np.sum(grad_output * self.x_normalized, axis=0)
        grad_beta = np.sum(grad_output, axis=0)

        # Gradient w.r.t. normalized input
        grad_x_normalized = grad_output * self.gamma

        # Gradient of normalization: chain rule through variance and mean
        N = self.d_model
        grad_var = np.sum(grad_x_normalized * (self.inputs - self.mean) * -0.5 * (self.var + self.eps) ** (-1.5), axis=-1, keepdims=True)
        grad_mean_from_var = grad_var * (-2.0 / N) * np.sum(self.inputs - self.mean, axis=-1, keepdims=True)
        grad_mean_direct = np.sum(grad_x_normalized * -1.0 / np.sqrt(self.var + self.eps), axis=-1, keepdims=True)
        grad_mean = grad_mean_direct + grad_mean_from_var

        # Gradient w.r.t. input
        grad_input = grad_x_normalized / np.sqrt(self.var + self.eps)
        grad_input += grad_var * 2.0 * (self.inputs - self.mean) / N
        grad_input += grad_mean / N

        # Update parameters
        self.gamma -= learning_rate * grad_gamma
        self.beta -= learning_rate * grad_beta

        return grad_input

class Transformer:
    def __init__(self, vocab_size, d_model, num_heads, d_ff, sequence_length, dropout_p=0.1):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, sequence_length)
        self.emb_dropout = Dropout(dropout_p)
        self.attention = Attention(d_model, num_heads)
        self.dropout1 = Dropout(dropout_p)
        self.dropout2 = Dropout(dropout_p)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.residual1 = ResidualConnection()
        self.norm1 = LayerNorm(d_model)
        self.residual2 = ResidualConnection()
        self.norm2 = LayerNorm(d_model)
        # Output projection: d_model -> vocab_size
        self.output_projection = Linear(d_model, vocab_size)

    def forward(self, x, training=True):
        assert x.ndim == 1, "Input should be a 1D array of token indices"
        # Embedding + Positional Encoding
        x = self.embedding.forward(x)
        x = self.positional_encoding.forward(x)
        x = self.emb_dropout.forward(x, training=training)
        self.pos_encoded = x

        # Attention block with residual and normaliation
        attn_output = self.attention.forward(x)
        attn_output = self.dropout1.forward(attn_output, training=training)
        x = self.residual1.forward(x, attn_output)
        x = self.norm1.forward(x)
        self.after_norm1 = x

        # Feed-forward block with residual and normalization
        ff_output = self.feed_forward.forward(x)
        ff_output = self.dropout2.forward(ff_output, training=training)
        x = self.residual2.forward(x, ff_output)
        x = self.norm2.forward(x)
        self.after_norm2 = x

        # Output projection to vocab size
        logits = self.output_projection.forward(x)
        return logits
    
    def backward(self, grad_output, learning_rate):
        # Gradient through output projection
        grad_after_norm2 = self.output_projection.backward(grad_output, learning_rate)
        
        # Gradient through norm2
        grad_residual2_input = self.norm2.backward(grad_after_norm2, learning_rate)
        
        # Split gradient through residual2 connection
        grad_x_before_ff, grad_ff_output = self.residual2.backward(grad_residual2_input)

        grad_ff_output = self.dropout2.backward(grad_ff_output)
        
        # Backward through feed-forward block
        grad_after_norm1 = self.feed_forward.backward(grad_ff_output, learning_rate)
        
        # Combine gradients from residual connection and feed-forward
        grad_after_norm1 = grad_after_norm1 + grad_x_before_ff
        
        # Gradient through norm1
        grad_residual1_input = self.norm1.backward(grad_after_norm1, learning_rate)
        
        # Split gradient through residual1 connection
        grad_pos_encoded, grad_attention_output = self.residual1.backward(grad_residual1_input)

        grad_attention_output = self.dropout1.backward(grad_attention_output)
        
        # Backward through attention block
        grad_attention_input = self.attention.backward(grad_attention_output, learning_rate)
        
        # Combine gradients from residual connection and attention
        grad_attention_input = grad_attention_input + grad_pos_encoded
        
        # Backward through positional encoding
        grad_embedding_output = self.positional_encoding.backward(grad_attention_input, learning_rate)

        grad_embedding_output = self.emb_dropout.backward(grad_embedding_output)
        
        # Backward through embedding (no gradient returned, just weight updates)
        self.embedding.backward(grad_embedding_output, learning_rate)
