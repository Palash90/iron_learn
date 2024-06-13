import numpy as np;

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
    return 1/(1 + np.exp(-x))

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

# network
net = NeuralNet(log_loss, log_loss_prime)
net.add(LinearLayer(2, 3))
net.add(ActivationLayer(sigmoid, sigmoid_prime))
net.add(LinearLayer(3, 1))
net.add(ActivationLayer(sigmoid, sigmoid_prime))

net.fit(x_train, y_train, epochs=100000, learning_rate=0.1)

# test
out = net.predict(x_train)
print(out)
