import time

import cupy as np
import numpy as onp
from transformer import Transformer
from simple_layers import Softmax, CrossEntropyLoss
from tokenizers import CharTokenizer, CorpusTokenizer
try:
    np.cuda.Device(0).use()
except Exception as e:
    print(f"GPU not found or error: {e}")

def generate_top_k(model, tokenizer, start_text, gen_length, k=5, temperature=1.0):
    # Ensure start_text isn't longer than what the model can handle
    if len(start_text) > model.positional_encoding.sequence_length:
        start_text = start_text[-model.positional_encoding.sequence_length:]
        
    input_tokens = np.array(tokenizer.encode(start_text))
    generated = start_text
    
    for _ in range(gen_length):
        logits = model.forward(input_tokens)[-1, :]
        logits = logits / (temperature + 1e-9)
        
        # Adjust k if vocab is smaller than k
        actual_k = min(k, tokenizer.vocab_size)
        top_k_indices = np.argpartition(logits, -actual_k)[-actual_k:]
        top_k_values = logits[top_k_indices]
        
        exp_values = np.exp(top_k_values - np.max(top_k_values))
        probs = exp_values / np.sum(exp_values)
        
        p_cpu = probs.get().astype('float64')
        p_cpu /= p_cpu.sum()
        idx_cpu = top_k_indices.get()
        
        chosen_index = onp.random.choice(idx_cpu, p=p_cpu)
        generated += tokenizer.decode([chosen_index])
        
        next_token_array = np.array([chosen_index], dtype=np.int32)
        input_tokens = np.append(input_tokens, next_token_array)
        if len(input_tokens) > model.positional_encoding.sequence_length:
            input_tokens = input_tokens[1:]
            
    return generated

# --- Main Logic ---

# 1. Load Data
with open('../../data/names.txt', 'r') as f:
    names = [line.strip() for line in f.readlines() if line.strip()]
    raw_text = "".join(names) + "\n"

# 2. Setup Tokenizer and Model
tokenizer = CharTokenizer(raw_text)
d_model = 128   # Slightly larger to handle more patterns
num_heads = 4
d_ff = 256
seq_len = 64 # 16 for names
learning_rate = 0.01 

model = Transformer(tokenizer.vocab_size, d_model, num_heads, d_ff, seq_len)
loss_fn = CrossEntropyLoss()
softmax_layer = Softmax()

# 3. Training Loop with Sliding Window
encoded_data = tokenizer.encode(raw_text)
epochs = 2000

print(f"Starting training on {len(names)} names...")

for epoch in range(epochs):
    total_loss = 0
    count = 0
    
    # Iterate through the data in steps of seq_len
    for i in range(0, len(encoded_data) - seq_len - 1, seq_len):
        inputs = encoded_data[i : i + seq_len]
        targets = encoded_data[i + 1 : i + seq_len + 1]
        
        # Forward
        logits = model.forward(inputs)
        probs = softmax_layer.forward(logits)
        
        # Loss & Backward
        loss = loss_fn.forward(probs, targets)
        total_loss += loss
        
        grad = loss_fn.backward()
        grad = softmax_layer.backward(grad)
        model.backward(grad, learning_rate)
        
        count += 1

        time.sleep(0.1)  # Slow down for better visualization
    
    if epoch % 50 == 0:
        avg_loss = total_loss / count
        print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f}", flush=True)
        # Sample a name to see progress
        sample = generate_top_k(model, tokenizer, "\n", 20, k=3)
        print(f"Sample: {sample.replace('\n', ' ')}")

# 4. Final Generation
print("\n--- Final Generated Names ---")
print(generate_top_k(model, tokenizer, "\n", 100, k=5, temperature=0.8))