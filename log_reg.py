import numpy as np

# Generate synthetic data (replace with your own dataset)
X = np.array([[0.5, 1.5], [1.0, 1.0], [1.5, 0.5], [3.0, 0.5], [2.0, 2.0], [1.0, 2.5]]) #np.random.rand(100, 2)  # Features (design matrix)
y = np.array([0,0,0,1,1,1]) #np.random.randint(0, 2, size=100)  # Binary outcome (0 or 1)

# Add bias term to X
X = np.hstack((np.ones((X.shape[0], 1)), X))
print(X)

# Initialize weights
w = np.array([0.0,0.0])

# Sigmoid function
def sigmoid(X, w):
    return 1 / (1 + np.exp(-X @ w))

# Compute predicted probabilities
p = sigmoid(X, w)

# Define loss function (e.g., cross-entropy)
def loss(y, p):
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

# Gradient descent (update weights)
learning_rate = 0.00001
for _ in range(100_000):
    gradient = X.T @ (sigmoid(X, w) - y) / len(y)
    w -= learning_rate * gradient
print(w)

# Threshold predictions
predictions = (p >= 0.5).astype(int)

# Evaluate model performance (accuracy, precision, recall, etc.)

# Your custom dataset and evaluation metrics will replace the synthetic data and placeholders.
