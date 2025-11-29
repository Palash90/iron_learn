import cupy as np
import json
import time # Added for benchmarking

# --- Layer Classes ---

class LinearLayer:
    def __init__(self, inputSize, outputSize):
        # Weights and biases are initialized on the GPU by CuPy
        self.weights = np.random.randn(inputSize, outputSize, dtype=np.float32)
        # Reshape biases to (1, outputSize) for broadcasting
        self.biases = np.random.randn(1, outputSize, dtype=np.float32) 
        
    def forward(self, input):
        # Input shape: (N, features) where N is the batch/sample size
        self.input = input
        # Matrix multiplication is highly optimized on GPU
        # (N, inputSize) @ (inputSize, outputSize) -> (N, outputSize)
        # Broadcasting adds (1, outputSize) biases to all N rows
        self.output = self.input @ self.weights + self.biases 
        return self.output
    
    def backward(self, error, lr):
        # error shape: (N, outputSize)
        # input shape: (N, inputSize)
        
        # input_error: (N, outputSize) @ (outputSize, inputSize) -> (N, inputSize)
        # Use np.dot for standard matrix multiplication (equivalent to @)
        input_error = error @ self.weights.T 
        
        # weights_error: (inputSize, N) @ (N, outputSize) -> (inputSize, outputSize)
        # np.dot(self.input.T, error) is correct
        weights_error = self.input.T @ error 
        
        # biases_error: sum error along the batch dimension (axis 0) to get (1, outputSize)
        # The total error from all samples is used to update the single bias vector
        biases_error = np.sum(error, axis=0, keepdims=True)
        
        self.weights -= lr * weights_error
        # Use biases_error which is the sum of the 'error' (dL/db) across the batch
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
        # Element-wise multiplication, which is fast on GPU
        return self.fPrime(self.input) * error
    
# --- Activation Functions ---
# CuPy's element-wise operations (ufuncs) are fast

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - np.tanh(x)**2

def sigmoid(x):
    # CuPy's built-in np.exp is optimized. We can simplify the implementation.
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    # Optimization: Use the calculated sigmoid output if possible, but calculating
    # it here is fine too.
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    # np.where is fast on GPU
    return np.where(x > 0, 1, 0)

# --- Loss Functions ---

def mse(y, y_hat):
    # Loss calculated over the entire batch/tensor is much faster
    return np.mean(np.power(y - y_hat, 2))

def mse_prime(y, y_hat):
    # Derivative is also calculated over the whole tensor
    # The division by y.size handles the averaging derivative
    return 2 * (y_hat - y) / y.size 

# --- Neural Network Class (Vectorized) ---

class NeuralNet:
    def __init__(self, loss, loss_prime):
        self.layers = []
        self.loss = loss
        self.loss_prime = loss_prime

    def add(self, layer):
        self.layers.append(layer)

    # --- VECTORIZED PREDICT ---
    def predict(self, input_data):
        # input_data is now expected to be a single (N, features) array
        # No loop over samples is needed!
        output = input_data 
        for layer in self.layers:
            output = layer.forward(output)
        return output

    # --- VECTORIZED FIT ---
    def fit(self, x_train, y_train, epochs, learning_rate):
        # x_train, y_train are now (N, features) and (N, targets) arrays
        samples = len(x_train) 
        
        # Ensure all data is on the GPU (it should be, but good practice)
        x_train = np.asarray(x_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)

        start_time = time.time()

        epoch_error = []

        for i in range(epochs):
            # 1. Forward Propagation (over the entire batch)
            output = x_train
            for layer in self.layers:
                output = layer.forward(output)

            # 2. Compute Loss (using vectorized loss function)
            err = self.loss(y_train, output)

            # 3. Backward Propagation (over the entire batch)
            error = self.loss_prime(y_train, output)
            for layer in reversed(self.layers):
                error = layer.backward(error, learning_rate)
            
            epoch_error.append([i, err])

            # 4. Display/Logging (less frequent printing is better for speed)
            if i == 0 or (i + 1) % 1000 == 0:
                # Synchronize to get accurate time/print results from GPU
                np.cuda.runtime.deviceSynchronize() 
                #print(f'Epoch {i+1}/{epochs} | Error: {err:.6f}')
        
        np.cuda.runtime.deviceSynchronize()
        end_time = time.time()
        with open('training_log_'+str(epochs)+'_'+str(learning_rate)+'.csv', 'w') as f:
            f.write('Epochs,Error\n')
            for epoch, error in epoch_error:
                f.write(f'{epoch},{error}\n')

        print(f"\nTraining completed in {end_time - start_time:.4f} seconds.")

def build_neural_net(features, outputs):
    net = NeuralNet(mse, mse_prime)

    net.add(LinearLayer(features, outputs))
#   net.add(ActivationLayer(relu, relu_prime))
    
#    net.add(LinearLayer(6, 6))
#    net.add(ActivationLayer(relu, relu_prime))
    
#    net.add(LinearLayer(6, outputs))
    
    return net

def run(epochs, learning_rate, data_field='linear'):
    # --- Data Preparation ---

    # Load and process data
    with open('data.json', 'r') as file:
        data = json.load(file)

    m = data[data_field]['m']
    n = data[data_field]['n']
    m_test = data[data_field]['m_test']

    # Rework data loading to be (N, features) and (N, targets), NOT (N, 1, features)
    # Load onto GPU as float32 for speed
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
    net.fit(x_train_norm, y_train_norm, epochs=epochs, learning_rate=learning_rate)

    # test
    y_pred = net.predict(x_test_norm) # y_pred shape is (m_test, 1)

    # Denormalize predictions
    y_pred_rescaled = y_pred * y_std + y_mean

    if data_field == 'linear':
        # Regression metrics for linear output
        mse_test = np.mean((y_pred_rescaled[:, 0] - y_test[:, 0])**2)
        mae_test = np.mean(np.abs(y_pred_rescaled[:, 0] - y_test[:, 0]))

        # Get final results onto CPU for print
        print(f"\nFinal Results after {epochs} epochs and learning rate {learning_rate}:")
        print(f"\nTest MSE: {mse_test.get():.4f}")
        print(f"Test MAE: {mae_test.get():.4f}")
    
    elif data_field == 'classification':
        # Classification accuracy
        correct = np.sum((y_pred_rescaled[:, 0] > 0.5) == (y_test[:, 0] > 0.5))
        accuracy = correct / m_test * 100

        print(f"\nFinal Results after {epochs} epochs and learning rate {learning_rate}:")
        print(f"\nTest Accuracy: {accuracy.get():.4f}% ({correct.get()} out of {m_test})")

if __name__ == "__main__":
    run(epochs=10000, learning_rate=0.005, data_field='linear')
