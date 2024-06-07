import numpy as np
import json

f = open('data.json')
data = json.load(f)

# Generate synthetic data (replace with your own dataset)
X = np.array(data["logistic"]["x"])
X = np.reshape(X, (data["logistic"]["m"],data["logistic"]["n"]))
y = np.array(data["logistic"]["y"]) 

# Initialize weights
w = np.zeros(data["logistic"]["n"])

print(X.shape, w.shape, y.shape)

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
lambda_ = 0.0
e = 100_000_00
e = 100_000_0
for _ in range(e):
    gradient = X.T @ (sigmoid(X, w) - y) / len(y)
    gradient += lambda_ * w / len(y) 
    w -= learning_rate * gradient
print(w)

yhat = sigmoid(X, w)
yhat = np.where(yhat > 0.5, 1.0, 0.0)
yhat = np.reshape(yhat, (data["logistic"]["m"]))

print("Predictions meet original", y == yhat)

