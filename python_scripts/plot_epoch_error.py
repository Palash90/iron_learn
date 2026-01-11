import json
import matplotlib.pyplot as plt
import numpy as np

def plot_refined_grokking(json_path, smoothing_window=50):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract [epoch, train_loss, val_loss]
    history = np.array(data.get('epoch_error', []))
    
    if history.size == 0:
        print("No training history found.")
        return

    epochs = history[:, 0]
    train_loss = history[:, 1]
    val_loss = history[:, 2]

    # Calculate moving average for validation loss to see the "hidden" trend
    val_smooth = np.convolve(val_loss, np.ones(smoothing_window)/smoothing_window, mode='valid')
    epochs_smooth = epochs[smoothing_window-1:]

    plt.figure(figsize=(14, 7))
    
    # Main Loss Curves
    plt.plot(epochs, train_loss, label='Train Loss (Raw)', color='#1f77b4', alpha=0.3)
    plt.plot(epochs, val_loss, label='Val Loss (Raw)', color='#d62728', alpha=0.3)
    
    # Smoothed Trend Line
    plt.plot(epochs_smooth, val_smooth, label=f'Val Loss ({smoothing_window}pt SMA)', color='#8c564b', linewidth=2)

    plt.yscale('log')
    plt.title(f"Experimental Progress: {data['name']} (Epoch {int(epochs[-1])})", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss (Log Scale)", fontsize=12)
    
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(loc='upper right')
    
    # Text box for current status
    stats_text = f"Current Val: {val_loss[-1]:.6f}\nTrain/Val Gap: {val_loss[-1]/train_loss[-1]:.1f}x"
    plt.text(0.02, 0.05, stats_text, transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

# Run this on your saved model
plot_refined_grokking('../model_outputs/grokking/model.json')