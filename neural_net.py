import numpy as np;
import json;

class LinearLayer:
    def __init__(self, inputSize, outputSize):
        self.weights = np.random.randn(inputSize, outputSize)
        self.biases = np.random.randn(1, outputSize)
        #self.weights = np.zeros((inputSize, outputSize))
        #self.biases = np.zeros((1, outputSize))
    
    def forward(self, input):
        self.input = input
        self.output = self.input @ self.weights + self.biases
        return self.output
    
    def backward(self, error, lr):
        input_error = np.dot(error, self.weights.T)
        weights_error = np.dot(self.input.T, error)
        
        self.weights -= lr * weights_error
        self.biases -= lr * error # Error for bias is independent of input
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
    
# activation function and its derivative
def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2

def sigmoid(x):
    out = np.empty_like(x)
    pos_mask = (x >= 0)
    out[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))
    neg_mask = (x < 0)
    exp_x = np.exp(x[neg_mask])
    out[neg_mask] = exp_x / (exp_x + 1)
    return out

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

# loss function and its derivative
def mse(y, y_hat):
    return np.mean(np.power(y-y_hat, 2))

def mse_prime(y, y_hat):
    return 2*(y_hat-y)/y.size

def log_loss(y, y_hat):
    return -y * np.log(y_hat) - (1 -y) * np.log(1 - y_hat)

def log_loss_prime(y, y_hat):
    return 2*(y_hat-y)/y.size

def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return np.where(x > 0, 1, 0)

class NeuralNet:
    def __init__(self, loss, loss_prime):
        self.layers = []
        self.loss = loss
        self.loss_prime = loss_prime

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)

        return result

    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate):
        # sample dimension first
        samples = len(x_train)

        # training loop
        for i in range(epochs):
            if(i % 100 == 0):
                print(f"Epoch {i}")
            err = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward(output)

                # compute loss (for display purpose only)
                err += self.loss(y_train[j], output)

                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate)

            # calculate average error on all samples
            err /= samples
            #print('epoch %d/%d   error=%f' % (i+1, epochs, err))


# training data
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# Open and read the JSON file
with open('data.json', 'r') as file:
    data = json.load(file)

data_field  = 'linear'  # Change this to 'logistic_2' to use the other dataset
m = data[data_field]['m']
n = data[data_field]['n']
m_test = data[data_field]['m_test']
x_train = np.array(data[data_field]['x']).reshape(m, 1, n)
x_mean = np.mean(x_train, axis=(0,1))
x_std = np.std(x_train, axis=(0,1))
x_train_norm = (x_train - x_mean) / (x_std + 1e-8)

y_train = np.array(data[data_field]['y']).reshape(m, 1, 1)
y_mean = np.mean(y_train)
y_std = np.std(y_train)
y_train_norm = (y_train - y_mean) / (y_std + 1e-8)

x_test = np.array(data[data_field]['x_test']).reshape(m_test, 1, n)
x_test_norm = (x_test - x_mean) / (x_std + 1e-8)

y_test = np.array(data[data_field]['y_test']).reshape(m_test, 1)
y_test_norm = (y_test - y_mean) / (y_std + 1e-8)


# network
net = NeuralNet(mse, mse_prime)
net.add(LinearLayer(n, 1))
#net.add(ActivationLayer(relu, relu_prime))
#net.add(LinearLayer(6, 1))
#net.add(ActivationLayer(sigmoid, sigmoid_prime))

net.fit(x_train_norm, y_train_norm, epochs=10000, learning_rate=0.1)

# test
y_pred = net.predict(x_test_norm)
y_pred = np.array(y_pred).reshape(m_test, 1)

# Denormalize predictions
y_pred_rescaled = y_pred * y_std + y_mean

# Compare to raw targets
print("Sample predictions vs actuals:")
for i in range(min(10, m_test)):
    print(f"Index {i}: predicted {y_pred_rescaled[i][0]:.2f}, actual {y_test[i][0]:.2f}")

# Regression metrics
mse_test = np.mean((y_pred_rescaled[:, 0] - y_test[:, 0])**2)
mae_test = np.mean(np.abs(y_pred_rescaled[:, 0] - y_test[:, 0]))
print(f"\nTest MSE: {mse_test:.4f}")
print(f"Test MAE: {mae_test:.4f}")
