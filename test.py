import numpy as np
import time
import json
from sklearn.linear_model import LinearRegression

def gradient_descent(x, y, alpha, num_iters):
    m, n = x.shape  # Number of training examples and features
    w = np.zeros((n, 1))  # Initialize weights

    start = time.time()


    for i in range(num_iters):
        # Calculate the gradient
        gradient = np.dot(x.T, (np.dot(x, w) - y)) / m
        # Update the weights
        w -= alpha * gradient

    end = time.time()
    print((end - start) * 1000, "milliseconds")


    return w

# Example usage:
# x: input features (shape: m x n, where m is the number of samples and n is the number of features)
# y: target values (shape: m x 1)
# alpha: learning rate
# num_iters: number of iterations
# w_optimal = gradient_descent(x, y, alpha=0.01, num_iters=1000)

f = open('data.json')
data = json.load(f)

x = data["x"]
x = np.array(x)
x = np.reshape(x, (data["m"], data["n"]))
y = data["y"]
y = np.array(y)
y = np.reshape(y, (data["m"],1))

alpha = 0.0001
num_iters= 10000
start = time.time()
w_optimal = gradient_descent(x, y, alpha, num_iters)
end = time.time()
print("My method took", (end - start) * 1000, "milliseconds")
print(np.reshape(w_optimal, data["n"]))

start = time.time()
model = LinearRegression().fit(x, y)
print(f"intercept: {model.intercept_}")
print(f"slope: {model.coef_}")
end = time.time()
print("SKLearn method took", (end - start) * 1000, "milliseconds")


