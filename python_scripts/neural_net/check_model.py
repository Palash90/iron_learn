import numpy as np

loaded = np.load('../200_final_output/checkpoint/checkpoint_epoch_1290201.npz', allow_pickle=True)
for key in loaded.files:
    print(f"{key}: {loaded[key].shape}")