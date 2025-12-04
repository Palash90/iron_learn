import cupy as np
import json
import time 
import networkx as nx
import matplotlib.pyplot as plt

def visualize_architecture(net):
    """
    Visualizes the architecture using layer names for labels.
    """
    layer_sizes = []
    layer_names = []

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

    plt.figure(figsize=(14, 8))
    nx.draw(G, pos, 
            node_size=500, 
            node_color=node_colors, 
            edge_color='gray', 
            with_labels=False, 
            arrows=True,
            alpha=0.9)
            
    for i, name in enumerate(layer_names):
        x_pos, y_pos = pos[f'{i+1}_0'] 
        plt.text(x_pos, y_pos + 1.5, name, 
                 fontsize=12, ha='center', fontweight='bold', color='darkslategray')
            
    plt.title("Neural Network Architecture Visualization")
    plt.axis('off')
    plt.show(block=False)

def plot_training_loss(loss_history):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Training Loss')
    plt.title('Model Training Progress')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show(block=False)

def plot_predictions(y_actual, y_predicted):
    if hasattr(y_actual, 'get'): y_actual = y_actual.get()
    if hasattr(y_predicted, 'get'): y_predicted = y_predicted.get()
    
    plt.figure(figsize=(10, 6))
    
    plt.scatter(range(len(y_actual)), y_actual, color='blue', alpha=0.5, label='Actual Data', s=10)
    
    plt.plot(range(len(y_predicted)), y_predicted, color='red', linewidth=2, label='Model Prediction')
    
    plt.title('Actual vs Predicted')
    plt.legend()
    plt.show(block=False)

class LinearLayer:
    def __init__(self, inputSize, outputSize):
        self.weights = np.random.randn(inputSize, outputSize, dtype=np.float32)
        self.biases = np.random.randn(1, outputSize, dtype=np.float32) 
        
    def forward(self, input):
        self.input = input
        self.output = self.input @ self.weights + self.biases 
        return self.output
    
    def backward(self, error, lr):
        input_error = error @ self.weights.T 
        weights_error = self.input.T @ error 
        
        biases_error = np.sum(error, axis=0, keepdims=True)
        
        self.weights -= lr * weights_error
        self.biases -= lr * biases_error 
        return input_error
    
class ActivationLayer:
    def __init__(self, activation, fPrime):
        self.activation = activation
        self.fPrime = fPrime
    
    def forward(self, input):
        self.input = input
        self.output = self.activation(self.input)
        return self.output
    
    def backward(self, error, lr):
        return self.fPrime(self.input) * error

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - np.tanh(x)**2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return np.where(x > 0, 1, 0)

def mse(y, y_hat):
    return np.mean(np.power(y - y_hat, 2))

def mse_prime(y, y_hat):
    return 2 * (y_hat - y) / y.size 

def binary_cross_entropy(y, y_hat):
    epsilon = 1e-12
    y_hat = np.clip(y_hat, epsilon, 1. - epsilon)
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

def binary_cross_entropy_prime(y, y_hat):
    epsilon = 1e-12
    y_hat = np.clip(y_hat, epsilon, 1. - epsilon)
    return ((1 - y) / (1 - y_hat) - y / y_hat) / y.size

class NeuralNet:
    def __init__(self, loss, loss_prime):
        self.layers = []
        self.loss = loss
        self.loss_prime = loss_prime
        self.layer_info = []

    def add(self, layer, name):
        self.layers.append(layer)
        self.layer_info.append({'layer': layer, 'name': name, 'type': layer.__class__.__name__})

    def predict(self, input_data):
        output = input_data 
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def fit(self, x_train, y_train, epochs, learning_rate):
        loss_history = [] 
        
        x_train = np.asarray(x_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)

        start_time = time.time()

        epoch_error = []

        for i in range(epochs):
            output = x_train
            for layer in self.layers:
                output = layer.forward(output)

            err = self.loss(y_train, output)

            error = self.loss_prime(y_train, output)
            loss_history.append(float(err))
            for layer in reversed(self.layers):
                error = layer.backward(error, learning_rate)
            
            epoch_error.append([i, err])

            if i == 0 or (i + 1) % 1000 == 0:
                np.cuda.runtime.deviceSynchronize() 
        
        np.cuda.runtime.deviceSynchronize()
        end_time = time.time()
        print(f"\nTraining completed in {end_time - start_time:.4f} seconds.")
        return loss_history

def build_neural_net(features, outputs):
    net = NeuralNet(binary_cross_entropy, binary_cross_entropy_prime)

    net.add(LinearLayer(features, 3), name = "Hidden Layer 1")
    net.add(ActivationLayer(relu, relu_prime), "Activation Layer")

    net.add(LinearLayer(3, 3), name = "Hidden Layer 2")
    net.add(ActivationLayer(relu, relu_prime), "Activation Layer")
    
    net.add(LinearLayer(3, outputs), name="Output Layer")
    net.add(ActivationLayer(sigmoid, sigmoid_prime), "Final Activation Layer")

    return net

def run(epochs, learning_rate, data_field='linear'):
    with open('../data.json', 'r') as file:
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

    net = build_neural_net(n, 1)

    print("Starting training...")
    history = net.fit(x_train_norm, y_train_norm, epochs=epochs, learning_rate=learning_rate)

    plot_training_loss(history)

    if n <= 20: 
        visualize_architecture(net)

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
        correct = np.sum((y_pred_rescaled[:, 0] > 0.5) == (y_test[:, 0] > 0.5))

        print(f"\nFinal Results after {epochs} epochs and learning rate {learning_rate}:")
        print(f"\nTest Correct Predictions: {correct.get()} out of {m_test}")

if __name__ == "__main__":
    run(epochs=10000, learning_rate=0.0001, data_field='logistic')
    input()
