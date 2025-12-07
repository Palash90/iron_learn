import cupy as np;

def sin(x):
    return np.sin(x)

def sin_prime(x):
    return np.cos(x)

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
