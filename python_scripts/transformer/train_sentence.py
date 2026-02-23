import time
import cupy as np
import numpy as onp
import pickle
import argparse
import os
from transformer import Transformer
from simple_layers import Softmax, CrossEntropyLoss
from tokenizers import CorpusTokenizer  # Using your word-level version

try:
    np.cuda.Device(0).use()
except Exception as e:
    print(f"GPU not found or error: {e}")

# Model checkpoint path
MODEL_PATH = "model.bin"
TOKENIZER_PATH = "tokenizer.pkl"
CHECKPOINT_PATH = "checkpoint.pkl"

def save_model(model, tokenizer, epoch, learning_rate):
    """Save model checkpoint"""
    checkpoint = {
        'model_state': model,
        'epoch': epoch,
        'learning_rate': learning_rate
    }
    with open(CHECKPOINT_PATH, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"Checkpoint saved: epoch {epoch}")

def load_checkpoint():
    """Load model checkpoint"""
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"No checkpoint found at {CHECKPOINT_PATH}")
        return None
    
    try:
        with open(CHECKPOINT_PATH, 'rb') as f:
            checkpoint = pickle.load(f)
        print(f"Checkpoint loaded: epoch {checkpoint['epoch']}")
        return checkpoint
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

def save_final_model(model):
    """Save final trained model"""
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"Final model saved to {MODEL_PATH}")

def load_model():
    """Load model for inference"""
    if not os.path.exists(MODEL_PATH):
        print(f"No model found at {MODEL_PATH}")
        return None
    
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {MODEL_PATH}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def save_tokenizer(tokenizer):
    """Save tokenizer"""
    with open(TOKENIZER_PATH, 'wb') as f:
        pickle.dump(tokenizer, f)

def load_tokenizer():
    """Load tokenizer"""
    if not os.path.exists(TOKENIZER_PATH):
        return None
    try:
        with open(TOKENIZER_PATH, 'rb') as f:
            tokenizer = pickle.load(f)
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return None

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

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Transformer sentence generation with save/load/resume')
    parser.add_argument('--train', action='store_true', help='Train the model and save to model.bin')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    parser.add_argument('--load', action='store_true', help='Load model and generate text (inference only)')
    
    args = parser.parse_args()
    
    # Check that exactly one mode is specified
    modes = sum([args.train, args.resume, args.load])
    if modes == 0:
        print("Error: Please specify one of --train, --resume, or --load")
        return
    if modes > 1:
        print("Error: Please specify only one of --train, --resume, or --load")
        return

    # 1. Load Data
    with open('../../data/sentences.txt', 'r') as f:
        raw_corpus = f.read()

    # 2. Setup Tokenizer
    if args.load:
        # For load mode, try to load saved tokenizer
        tokenizer = load_tokenizer()
        if tokenizer is None:
            print("No saved tokenizer found. Creating new tokenizer...")
            tokenizer = CorpusTokenizer(raw_corpus)
    else:
        # For train/resume, always create/use fresh tokenizer
        tokenizer = CorpusTokenizer(raw_corpus)
        save_tokenizer(tokenizer)

    # 3. Model Hyperparameters
    d_model = 256    # Increased for word embeddings
    num_heads = 8
    d_ff = 512
    seq_len = 32     # 32 words is a lot of context for a sentence
    learning_rate = 0.05
    epochs = 10000
    start_epoch = 0

    # Initialize or load model
    if args.train:
        print("=== TRAINING MODE ===")
        model = Transformer(tokenizer.vocab_size, d_model, num_heads, d_ff, seq_len)
        loss_fn = CrossEntropyLoss()
        softmax_layer = Softmax()
        
    elif args.resume:
        print("=== RESUME TRAINING MODE ===")
        checkpoint = load_checkpoint()
        if checkpoint is None:
            print("No checkpoint found. Starting fresh training...")
            model = Transformer(tokenizer.vocab_size, d_model, num_heads, d_ff, seq_len)
            loss_fn = CrossEntropyLoss()
            softmax_layer = Softmax()
        else:
            model = checkpoint['model_state']
            start_epoch = checkpoint['epoch'] + 1
            learning_rate = checkpoint['learning_rate'] * 0.99  # Adjust learning rate for resumed training
            loss_fn = CrossEntropyLoss()
            softmax_layer = Softmax()
            print(f"Resuming from epoch {start_epoch} with learning rate {learning_rate:.6f}")
    
    elif args.load:
        print("=== INFERENCE MODE ===")
        model = load_model()
        tokenizer = load_tokenizer()
        if model is None or tokenizer is None:
            print("Error: Cannot load model or tokenizer. Make sure to train first with --train or --resume")
            return
        
        # Generate text in inference mode
        print("\n--- Generated Text ---")
        print(generate_sentence(model, tokenizer, "Artificial intelligence", 20, k=5))
        print(generate_sentence(model, tokenizer, "The", 15, k=5))
        print(generate_sentence(model, tokenizer, "Machine learning", 15, k=5))
        return

    # 4. Training (skip for load mode)
    if args.train or args.resume:
        # We extract lines to train on structured sentences
        lines = [line.strip() for line in raw_corpus.splitlines() if line.strip()]

        print(f"Starting training on {len(lines)} sentences. Vocab size: {tokenizer.vocab_size}")
        print(f"Starting from epoch {start_epoch}")

        t0 = time.time()
        for epoch in range(start_epoch, epochs):
            learning_rate *= 0.99  # Decay learning rate
            learning_rate = max(learning_rate, 1e-5)  # Minimum learning rate
            total_loss = 0
            count = 0
            
            onp.random.shuffle(lines) # Shuffle sentences each epoch

            
            
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

            time.sleep(2)  # Small sleep to prevent GPU overheating in this simple implementation
            
            if epoch % 10 == 0:
                avg_loss = total_loss / count
                t1 = time.time()
                print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f} | Learning Rate: {learning_rate:.6f} | Time: {t1 - t0:.2f}s", flush=True)
                t0 = t1
                # Sample generation
                print(f"Sample: {generate_sentence(model, tokenizer, 'The', 10)}", flush=True)
                
                # Save checkpoint every 10 epochs
                save_model(model, tokenizer, epoch, learning_rate)

        # 5. Final Generation and Save
        print("\n--- Final Generated Text ---")
        print(generate_sentence(model, tokenizer, "Artificial intelligence", 20, k=5))
        
        # Save final model
        save_final_model(model)

if __name__ == "__main__":
    main()