from datetime import datetime
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
    print("GPU found and will be used for training.")
except Exception as e:
    print(f"GPU not found or error: {e}")

# Model checkpoint path
MODEL_PATH = "agent.bin"
TOKENIZER_PATH = "agent.pkl"
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

def generate_answer(model, tokenizer, start_text, gen_length, k=1, temperature=0.1):
    # 1. Standard Setup
    words = start_text.split()
    input_ids = [tokenizer.char_to_id[w] for w in words if w in tokenizer.char_to_id]
    
    # Use a list to store generated IDs for easier appending
    generated_ids = list(input_ids)

    obs_text = ""  # To store the latest observation for display
    
    for _ in range(gen_length):
        # Prepare context (handle sequence length)
        curr_input = np.array(generated_ids, dtype=np.int32)
        if len(curr_input) > model.positional_encoding.sequence_length:
            curr_input = curr_input[-model.positional_encoding.sequence_length:]
            
        # Forward pass - take the last logit
        logits = model.forward(curr_input, training=False)[-1, :]
        
        # --- SAMPLING LOGIC ---
        # For Agents, k=1 (Greedy) is much safer than k=5
        chosen_index = int(np.argmax(logits)) 
        
        # 2. BREAK CONDITIONS
        if chosen_index == tokenizer.char_to_id.get("<|end|>", -1):
            break
            
        generated_ids.append(chosen_index)
        current_text = " ".join([tokenizer.id_to_char[idx] for idx in generated_ids])

        # 3. THE TOOL HOOK (The "Pause" button)
        # Check if the model just finished writing an action: e.g., "CALC( 2 + 2 )"
        if ")" in tokenizer.id_to_char[chosen_index] and "Action:" in current_text:
            
            # --- EXECUTE PYTHON TOOL ---
            if "CALC(" in current_text:
                # Extract content between CALC( and )
                print("Tool CALC() called. Evaluating expression for observation.")
                expr = current_text.split("CALC(")[-1].split(")")[0]
                print(f"Evaluating expression: {expr}")
                try:
                    # BE CAREFUL with eval in production; use a safe math parser if possible
                    observation = str(eval(expr.replace(' ', '')))
                except:
                    observation = "Error"
            elif "GET_TIME()" in current_text:
                import datetime
                print("Tool GET_TIME() called. Injecting current time as observation.")
                observation = datetime.datetime.now().strftime("%H:%M")
            else:
                observation = "Unknown Tool"

            # 4. INJECT THE OBSERVATION
            # Manually append the result to the conversation
            print(f"Injected Observation: {observation}")
            obs_text = f" Answer: {observation}"
    
            continue

    return " ".join([tokenizer.id_to_char[idx] for idx in generated_ids]) + obs_text
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
    with open('../../data/agent_data.txt', 'r') as f:
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
    d_model = 128    # Increased for word embeddings
    num_heads = 16
    d_ff = 512
    seq_len = 32
    learning_rate = 0.01
    epochs = 840
    start_epoch = 0
    dropout_p = 0.1

    # Initialize or load model
    if args.train:
        print("=== TRAINING MODE ===")
        model = Transformer(tokenizer.vocab_size, d_model, num_heads, d_ff, seq_len, dropout_p)
        loss_fn = CrossEntropyLoss()
        softmax_layer = Softmax()
        
    elif args.resume:
        print("=== RESUME TRAINING MODE ===")
        checkpoint = load_checkpoint()
        if checkpoint is None:
            print("No checkpoint found. Starting fresh training...")
            model = Transformer(tokenizer.vocab_size, d_model, num_heads, d_ff, seq_len, dropout_p)
            loss_fn = CrossEntropyLoss()
            softmax_layer = Softmax()
        else:
            model = checkpoint['model_state']
            start_epoch = checkpoint['epoch'] + 1
            learning_rate = checkpoint['learning_rate'] * 0.995  # Adjust learning rate for resumed training
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
        
        k = 20
        temp = 0
        gen_length = 16
        sample = 20

        print(f"Generating {sample} tokens with k={k}, temperature={temp}, and gen_length={gen_length}:\n")

        user_input = ""

        while user_input != "exit":
            user_input = input("Enter a prompt (or 'exit' to quit): ")
            if user_input == "exit":
                break
            print(generate_answer(model, tokenizer, user_input, gen_length, temperature=temp, k=k))
        
        return

    # 4. Training (skip for load mode)
    if args.train or args.resume:
        print("D_Model:", d_model, "Num Heads:", num_heads, "D_FF:", d_ff, "Seq Len:", seq_len)

        lines = [line.strip() for line in raw_corpus.splitlines() if line.strip()]

        split_point = int(len(lines) * 0.2)
        trainining_lines = lines[split_point:]
        validation_lines = lines[:split_point]
        
        print(f"Starting training on {len(trainining_lines)} sentences, keeping {len(validation_lines)} for validation. Vocab size: {tokenizer.vocab_size}")
        print(f"Starting from epoch {start_epoch}")

        t0 = time.time()
        for epoch in range(start_epoch, epochs):
            learning_rate *= 0.995  # Decay learning rate
            learning_rate = max(learning_rate, 1e-5)  # Minimum learning rate
            total_loss = 0
            count = 0
            
            onp.random.shuffle(trainining_lines) # Shuffle sentences each epoch

            for line in trainining_lines:
                # Encode with SOS/EOS and Padding
                tokens = tokenizer.encode_sentence(line, seq_len + 1)
                
                # X: [SOS, W1, W2...] -> Y: [W1, W2, ..., EOS]
                inputs = tokens[:-1]
                targets = tokens[1:]
                
                # Forward pass
                logits = model.forward(inputs, training=True)
                probs = softmax_layer.forward(logits)
                
                # Backward pass
                loss = loss_fn.forward(probs, targets)
                total_loss += loss
                
                grad = loss_fn.backward()
                #grad = softmax_layer.backward(grad)
                model.backward(grad, learning_rate)
                
                count += 1

            time.sleep(0.1)  # Small sleep to prevent GPU overheating in this simple implementation
            
            if epoch % 10 == 0:
                avg_loss = total_loss / count
                t1 = time.time()
                
                # Instead of one random line, use a small batch for stability
                val_subset = validation_lines
                val_losses = []
                for v_line in val_subset:
                    v_tokens = tokenizer.encode_sentence(v_line, seq_len + 1)
                    v_logits = model.forward(v_tokens[:-1], training=False)
                    v_probs = softmax_layer.forward(v_logits)
                    val_losses.append(loss_fn.forward(v_probs, v_tokens[1:]).get())
                
                avg_val_loss = onp.mean(onp.array(val_losses))

                print(f"{datetime.now()} Epoch {epoch} | Avg Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Learning Rate: {learning_rate:.6f} | Time: {t1 - t0:.2f}s", flush=True)
                t0 = t1
                # Sample generation (with dropout during training)
                print(f"Sample: {generate_answer(model, tokenizer, 'User: time please', 10, temperature=0)}", flush=True)

                print(f"Sample: {generate_answer(model, tokenizer, 'User: 4 subtract 2 = ?', 10, temperature=0)}", flush=True)

                print(f"Sample: {generate_answer(model, tokenizer, 'User: current time', 10, temperature=0)}", flush=True)

                print(f"Sample: {generate_answer(model, tokenizer, 'User: 3 sum 2 = ?', 10, temperature=0)}", flush=True)

                print(f"Sample: {generate_answer(model, tokenizer, 'User: 40 minus 27 = ?', 10, temperature=0)}", flush=True)
                
                # Save checkpoint every 10 epochs
                save_model(model, tokenizer, epoch, learning_rate)

        # Save final model
        save_final_model(model)

if __name__ == "__main__":
    main()