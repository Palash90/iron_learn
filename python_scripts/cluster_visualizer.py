import json
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def visualize_clusters(model_path, metadata_path):
    # 1. Load Metadata for character mapping
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    itos = metadata["itos"]
    vocab_size = metadata["vocab_size"]

    # 2. Extract Weights from model.json
    # Note: We use regex because the file contains trailing history data
    # that makes standard json.load() fail.
    with open(model_path, "r") as f:
        content = f.read()

    match = re.search(
        r'\{[^{]*"name":\s*"Input"[^{]*"weights":\s*\[([^\]]*)\]', content, re.DOTALL
    )
    if not match:
        print("Could not find Input layer weights.")
        return

    weights_raw = [float(x.strip()) for x in match.group(1).split(",") if x.strip()]

    # 3. Reshape and Process
    # Input is 108 (4 positions * 27 chars) -> Hidden 216
    W = np.array(weights_raw).reshape(108, 216)

    # Reshape to (Position, Character, Hidden_Dim)
    # 108 = 4 positions * 27 characters
    char_reps = W.reshape(4, 27, 216)

    # Average across the 4 context positions to get a general char representation
    avg_reps = np.mean(char_reps, axis=0)

    # 4. Dimensionality Reduction (PCA)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(avg_reps)

    # 5. Plotting
    plt.figure(figsize=(10, 8))
    plt.scatter(coords[:, 0], coords[:, 1], color="skyblue", s=100, edgecolors="navy")

    for i in range(vocab_size):
        char = itos[str(i)]
        # Highlight vowels in red to see if they cluster
        color = "red" if char in "aeiou" else "black"
        plt.annotate(
            char,
            (coords[i, 0], coords[i, 1]),
            fontsize=14,
            fontweight="bold",
            color=color,
            xytext=(5, 5),
            textcoords="offset points",
        )

    plt.title("Character Clusters (5-gram Model Input Weights)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig("char_clusters.png")
    print("Visualization saved as char_clusters.png")


if __name__ == "__main__":
    visualize_clusters("../model_outputs/5-gram/model.json", "../model_outputs/5-gram/n_gram_metadata.json")
