import json
import numpy as np
import matplotlib.pyplot as plt

def plot_weight_distribution(json_path, layer_idx=1):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    weights = np.array(data['layers'][layer_idx]['weights'])
    
    plt.figure(figsize=(10, 6))
    plt.hist(weights.flatten(), bins=100, color='crimson', alpha=0.7)
    plt.yscale('log') # Log scale is essential to see the "expert" weights
    plt.title(f'Layer {layer_idx} Weight Distribution (Log Scale)')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency (Count)')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.show()

if __name__ == "__main__":
    plot_weight_distribution('../model_outputs/grokking/model.json', 0)