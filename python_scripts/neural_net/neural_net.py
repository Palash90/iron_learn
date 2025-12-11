import json
import time
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd 
import numpy
from layers import *
from builder import *

np.cuda.runtime.setDevice(0)

def visualize_network(net):
    """
    Visualizes the architecture using layer names for labels.
    """
    layer_sizes = []
    layer_names = ["Input"]

    linear_layers = [info for info in net.layer_info if isinstance(info['layer'], LinearLayer)]

    if not linear_layers:
        print("No Linear Layers found to visualize.")
        return
    
    first_layer = linear_layers[0]['layer']
    layer_sizes.append(first_layer.weights.shape[0])

    for info in linear_layers:
        layer = info['layer']
        layer_sizes.append(layer.weights.shape[1])
        layer_names.append(info['name'])
            
    print(f"Detected Layer Sizes: {layer_sizes}")
    print(f"Detected Layer Names: {layer_names}")

    G = nx.DiGraph()
    subset_sizes = layer_sizes
    pos = {}
    node_colors = []
    
    v_spacing = 1.0
    h_spacing = 2.5
    
    for i, layer_size in enumerate(subset_sizes):
        layer_top = (layer_size - 1) / 2.0 * v_spacing
        for j in range(layer_size):
            node_id = f'{i}_{j}'
            G.add_node(node_id, layer=i)
            pos[node_id] = (i * h_spacing, layer_top - j * v_spacing)
            
            if i == 0: color = 'gold'
            elif i == len(subset_sizes) - 1: color = 'salmon'
            else: color = 'skyblue'
            node_colors.append(color)
            
            if i > 0:
                prev_layer_size = subset_sizes[i-1]
                for k in range(prev_layer_size):
                    G.add_edge(f'{i-1}_{k}', node_id)

    text_color = 'white'
    dark_navy = '#0A0A1F'

    fig = plt.figure(figsize=(14, 8), facecolor=dark_navy) 
    ax = fig.add_subplot(111)
    ax.set_facecolor(dark_navy)

    nx.draw(G, pos, 
            node_size=500, 
            node_color=node_colors, 
            edge_color='silver', 
            with_labels=False, 
            arrows=True,
            alpha=0.9)
    
    
    
    if layer_names:
        input_layer_name = layer_names[0]
        x_pos_in, y_pos_in = pos[f'0_0'] 
        
        plt.text(x_pos_in, y_pos_in + 1.5, input_layer_name, 
                 fontsize=12, ha='center', fontweight='bold', color=text_color)
            
    for i, name in enumerate(layer_names[1:]):
        graph_index = i + 1
        x_pos, y_pos = pos[f'{graph_index}_0'] 
        plt.text(x_pos, y_pos + 1.5, name, 
                 fontsize=12, ha='center', fontweight='bold', color=text_color)
            
    fig.set_facecolor(dark_navy)
    plt.axis('off')
    plt.show(block=False)

def run(epochs, learning_rate, data_field='linear'):
    with open('../../data.json', 'r') as file:
        data = json.load(file)

    m = data[data_field]['m']
    n = data[data_field]['n']
    m_test = data[data_field]['m_test']

    x_train = np.array(data[data_field]['x'], dtype=np.float32).reshape(m, n) 
    x_mean = np.mean(x_train, axis=0)
    x_std = np.std(x_train, axis=0)
    x_train_norm = (x_train - x_mean) / (x_std + 1e-8)

    y_train = np.array(data[data_field]['y'], dtype=np.float32).reshape(m, 1)
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    y_train_norm = (y_train - y_mean) / (y_std + 1e-8)

    x_test = np.array(data[data_field]['x_test'], dtype=np.float32).reshape(m_test, n)
    x_test_norm = (x_test - x_mean) / (x_std + 1e-8)

    y_test = np.array(data[data_field]['y_test'], dtype=np.float32).reshape(m_test, 1)

    net = build_neural_net(n, 1, 6, tanh, tanh_prime)
    visualize_network(net)

    def hook(net, epoch):
        pass

    print("Starting training...")
    net.fit(x_train_norm, y_train_norm, epochs=epochs, epoch_offset= 0, learning_rate=learning_rate, hook = hook)

    # test
    y_pred = net.predict(x_test_norm)

    # Denormalize predictions
    y_pred_rescaled = y_pred # * y_std + y_mean

    if data_field == 'linear':
        # Regression metrics for linear output
        mse_test = np.mean((y_pred_rescaled[:, 0] - y_test[:, 0])**2)
        mae_test = np.mean(np.abs(y_pred_rescaled[:, 0] - y_test[:, 0]))

        # Get final results onto CPU for print
        print(f"\nFinal Results after {epochs} epochs and learning rate {learning_rate}:")
        print(f"\nTest MSE: {mse_test.get():.4f}")
        print(f"Test MAE: {mae_test.get():.4f}")
    
    elif data_field == 'logistic':
        # Classification accuracy
        correct = np.sum((y_pred_rescaled[:, 0] > 0.5) == (y_test[:, 0] > 0.5))
        accuracy = correct / m_test * 100

        print(f"\nFinal Results after {epochs} epochs and learning rate {learning_rate}:")
        print(f"\nTest Accuracy: {accuracy.get():.4f}% ({correct.get()} out of {m_test})")
    else:
        for pred, actual in zip(y_pred_rescaled, y_test):
            print(f"Y: {actual}, P: {pred}")

        print(f"\nFinal Results after {epochs} epochs and learning rate {learning_rate}:")


def load_data_from_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)
        
        X = df[['x', 'y']].values.astype(np.float32)
        
        Y = df[['pixel_value']].values.astype(np.float32)
        
        max_x_index = df['x'].max()
        max_y_index = df['y'].max()

        norm_factors = (max_x_index, max_y_index)
        
        X[:, 0] = X[:, 0] / max_x_index 
        X[:, 1] = X[:, 1] / max_y_index
        
        print(f"Loaded {len(X)} data points.")
        print(f"X shape (Input): {X.shape}, Y shape (Target): {Y.shape}")
        
        return X, Y, norm_factors
    except FileNotFoundError:
        print(f"❌ Error: CSV file not found at '{csv_path}'. Run the conversion first!")
        return None, None
    except Exception as e:
        print(f"❌ An error occurred during data loading: {e}")
        return None, None


def draw_predictions_scatter(co_ordinates, epoch, width, height, values):
    print(f"\n🎨 Generating scatter plot visualization at epoch {epoch}...")

    print(values.shape)

    x_coords = np.asnumpy(co_ordinates[:, 0]) 
    y_coords = np.asnumpy(co_ordinates[:, 1]) 
    
    x_coords_denorm = x_coords * width
    y_coords_denorm = y_coords * height
    
    pixel_values = np.asnumpy(values[:, 0])
    
    fig, ax = plt.subplots(figsize=(5, 5)) 

    ax.scatter(
        x_coords_denorm, 
        y_coords_denorm, 
        c=pixel_values, 
        cmap='gray_r', 
        vmin=0, 
        vmax=1,
        s=1, 
        marker='s' 
    )

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0) 
    ax.set_aspect('equal')

    ax.axis('off')

    if epoch != 'ORIGINAL':
        plt.title(f'Image drawn on {epoch}-th try')
    else:
        plt.title('Original Image')
    # Save the plot
    file_name = "output/image/plot_"+ str(epoch) +".png"
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"🖼️ Saved scatter plot: {file_name} successfully!")

def image_reconstruction():
    X_train, Y_train, norm_factors = load_data_from_csv("../image_inputs/pixel_data_200.csv")

    IMAGE_WIDTH = norm_factors[0] + 1
    IMAGE_HEIGHT = norm_factors[1] + 1
    CHECKPOINT = 1000
    EPOCHS = 200001
    LEARNING_RATE = 0.00005
    EPOCH_OFFSET = 0 
    RESUME_FILE = ''
    TIME_CHECK = 1000
    LAST_EPOCH = 0

    X_train = np.asarray(X_train)
    # X_train = 2 * X_train - 1

    Y_train = np.asarray(Y_train)

    draw_predictions_scatter(X_train, "ORIGINAL", IMAGE_WIDTH, IMAGE_HEIGHT, Y_train)

    epoch_start_time =  time.time()

    def epoch_hook(net, epoch):
        global epoch_start_time, LAST_EPOCH

        if epoch % CHECKPOINT == 0:
            pass # net.save_weights(f'output/checkpoint/checkpoint_epoch_{epoch+EPOCH_OFFSET+1}.npz')

        if epoch % 1000 == 0:
            (f"\n\t\tDrawing at epoch {epoch}")
            predictions = net.predict(X_train)
            draw_predictions_scatter(X_train, epoch + EPOCH_OFFSET, IMAGE_WIDTH, IMAGE_HEIGHT, predictions)
        
        if epoch % TIME_CHECK == 0:
            epoch_end_time = time.time()
            print(f"Elapsed time {epoch_end_time - epoch_start_time: .2f} seconds for {TIME_CHECK} iterations {LAST_EPOCH} - {epoch}")
            epoch_start_time = time.time()
            LAST_EPOCH = epoch

    if X_train is not None:
        INPUT_FEATURES = X_train.shape[1] 
        OUTPUT_NODES = Y_train.shape[1]
        # net = build_neural_net(INPUT_FEATURES, OUTPUT_NODES, 50, tanh, tanh_prime)
        net = build_siren_net(INPUT_FEATURES, OUTPUT_NODES, 50)
        
        if net.load_weights(RESUME_FILE):
             print(f"Resuming training from {RESUME_FILE}")
        else:
             print("Starting training from scratch.")

        print(f"\n🚀 Starting training for {EPOCHS} epochs...")
        net.fit(X_train, Y_train, epochs=EPOCHS, epoch_offset=EPOCH_OFFSET, learning_rate=LEARNING_RATE, hook=epoch_hook)  

if __name__ == "__main__":
    run(10000, 0.001, 'neural_network')