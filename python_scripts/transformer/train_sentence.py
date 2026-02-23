import time
import cupy as np
import numpy as onp
from transformer import Transformer
from simple_layers import Softmax, CrossEntropyLoss
from tokenizers import CorpusTokenizer  # Using your word-level version

try:
    np.cuda.Device(0).use()
except Exception as e:
    print(f"GPU not found or error: {e}")

def generate_sentence(model, tokenizer, start_text, gen_length, k=5, temperature=1.0):
    # Prepare the initial tokens (Words, not characters)
    words = start_text.split()
    # If starting fresh, you could use [tokenizer.sos_id]
    input_ids = [tokenizer.char_to_id[w] for w in words if w in tokenizer.char_to_id]
    
    if not input_ids: # Fallback to SOS if no valid start text
        input_ids = [tokenizer.sos_id]
        
    input_tokens = np.array(input_ids, dtype=np.int32)
    generated_words = words
    
    for _ in range(gen_length):
        # Clip context to model's seq_len
        curr_input = input_tokens
        if len(curr_input) > model.positional_encoding.sequence_length:
            curr_input = curr_input[-model.positional_encoding.sequence_length:]
            
        logits = model.forward(curr_input)[-1, :]
        logits = logits / (temperature + 1e-9)
        
        # Top-k sampling
        actual_k = min(k, tokenizer.vocab_size)
        top_k_indices = np.argpartition(logits, -actual_k)[-actual_k:]
        top_k_values = logits[top_k_indices]
        
        exp_values = np.exp(top_k_values - np.max(top_k_values))
        probs = (exp_values / np.sum(exp_values)).get().astype('float64')
        probs /= probs.sum()
        
        chosen_index = onp.random.choice(top_k_indices.get(), p=probs)
        
        # Stop if model predicts EOS
        if chosen_index == tokenizer.eos_id:
            break
            
        # Decode and append word
        word = tokenizer.id_to_char[int(chosen_index)]
        if chosen_index > 2: # Skip special tokens like <PAD>
            generated_words.append(word)
            
        input_tokens = np.append(input_tokens, np.array([chosen_index], dtype=np.int32))
            
    return " ".join(generated_words)

# --- Main Logic ---

# 1. Load Data
with open('../../data/sentences.txt', 'r') as f:
    raw_corpus = f.read()

# 2. Setup Tokenizer
tokenizer = CorpusTokenizer(raw_corpus)

# 3. Model Hyperparameters
d_model = 256    # Increased for word embeddings
num_heads = 8
d_ff = 512
seq_len = 32     # 32 words is a lot of context for a sentence
learning_rate = 0.01

model = Transformer(tokenizer.vocab_size, d_model, num_heads, d_ff, seq_len)
loss_fn = CrossEntropyLoss()
softmax_layer = Softmax()

# 4. Training
# We extract lines to train on structured sentences
lines = [line.strip() for line in raw_corpus.splitlines() if line.strip()]
epochs = 10000

print(f"Starting training on {len(lines)} sentences. Vocab size: {tokenizer.vocab_size}")

for epoch in range(epochs):
    learning_rate *= 0.99  # Decay learning rate
    total_loss = 0
    count = 0
    
    onp.random.shuffle(lines) # Shuffle sentences each epoch

    t0 = time.time()
    
    for line in lines:
        # Encode with SOS/EOS and Padding
        tokens = tokenizer.encode_sentence(line, seq_len + 1)
        
        # X: [SOS, W1, W2...] -> Y: [W1, W2, ..., EOS]
        inputs = tokens[:-1]
        targets = tokens[1:]
        
        # Forward pass
        logits = model.forward(inputs)
        probs = softmax_layer.forward(logits)
        
        # Backward pass
        loss = loss_fn.forward(probs, targets)
        total_loss += loss
        
        grad = loss_fn.backward()
        grad = softmax_layer.backward(grad)
        model.backward(grad, learning_rate)
        
        count += 1

    time.sleep(20)  # Small sleep to prevent GPU overheating in this simple implementation
    
    if epoch % 10 == 0:
        avg_loss = total_loss / count
        t1 = time.time()
        print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f} | Learning Rate: {learning_rate:.6f} | Time: {t1 - t0:.2f}s")
        t0 = t1
        # Sample generation
        print(f"Sample: {generate_sentence(model, tokenizer, 'The', 10)}")

# 5. Final Generation
print("\n--- Final Generated Text ---")
print(generate_sentence(model, tokenizer, "Artificial intelligence", 20, k=5))