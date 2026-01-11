import json
import numpy as np
import matplotlib.pyplot as plt

def visualize_1d_weights(model_path, p=13):
    with open(model_path, 'r') as f:
        data = json.load(f)
    
    # 1. Extract the 1D vector
    # Adjust 'weights' key based on your JSON structure
    raw_vec = np.array(data['layers'][0]['weights']) 
    
    # 2. Reshape it
    # We assume shape is (Inputs, Hidden_Units)
    # If the length is 19400, and inputs are 194, then hidden units = 100
    hidden_units = len(raw_vec) // (p * 2)
    weights_matrix = raw_vec.reshape((p * 2, hidden_units))
    
    # 3. Split into Input A (first 97) and Input B (next 97)
    weights_a = weights_matrix[:p, :]
    weights_b = weights_matrix[p:, :]
    
    plt.figure(figsize=(15, 5))
    
    # Subplot 1: Raw Weights for 'a'
    plt.subplot(1, 3, 1)
    plt.imshow(weights_a, aspect='auto', cmap='magma')
    plt.title("Weights for Input A")
    plt.ylabel("Number (0-96)")
    plt.xlabel("Hidden Neurons")

    # Subplot 2: The "Grokking Check" (Correlation)
    # This shows if numbers that are 'close' mathematically are treated similarly
    plt.subplot(1, 3, 2)
    corr = np.corrcoef(weights_a)
    plt.imshow(corr, cmap='coolwarm')
    plt.title("Similarity of Numbers")
    
    # Subplot 3: Fourier Transform (The Logic Check)
    # If grokked, you will see a bright dot at specific frequencies
    plt.subplot(1, 3, 3)
    fourier = np.abs(np.fft.fft(weights_a, axis=0))
    plt.imshow(fourier[:p//2, :], aspect='auto', cmap='viridis')
    plt.title("Frequency Analysis (Grokking Signal)")
    
    plt.tight_layout()
    plt.show()

visualize_1d_weights('../model_outputs/grokking/model.json')