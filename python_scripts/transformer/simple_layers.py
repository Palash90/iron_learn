import cupy as np

try:
    np.cuda.Device(0).use()
except Exception as e:
    print(f"GPU not found or error: {e}")

class Linear:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)

    def forward(self, x):
        self.inputs = x
        return np.dot(x, self.weight)
    
    def backward(self, grad_output, learning_rate):
        grad_input = np.dot(grad_output, self.weight.T)
        grad_weight = np.dot(self.inputs.T, grad_output)
        self.weight -= learning_rate * grad_weight
        return grad_input
    
class ReLU:
    def __init__(self):
        self.input = None

    def forward(self, x):
        self.input = x
        return np.maximum(0, x)
    
    def backward(self, grad_output):
        grad_output[self.input <= 0] = 0
        return grad_output

class Softmax:
    def __init__(self):
        self.output = None

    def forward(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        self.output = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        return self.output
    
    def backward(self, grad_output):
        grad_input = self.output * (grad_output - np.sum(grad_output * self.output, axis=-1, keepdims=True))
        return grad_input
    
class CrossEntropyLoss:
    def __init__(self):
        self.predictions = None
        self.targets = None

    def forward(self, predictions, targets):
        self.predictions = predictions
        self.targets = targets
        m = self.targets.shape[0]
        p = self.predictions[np.arange(m), self.targets]
        p = np.clip(p, 1e-15, 1 - 1e-15)  # Avoid log(0)
        log_likelihood = -np.log(p + 1e-15)
        loss = np.sum(log_likelihood) / m
        return loss
    
    def backward(self):
        m = self.targets.shape[0]
        grad = self.predictions.copy()
        grad[np.arange(m), self.targets] -= 1
        grad /= m
        return grad
