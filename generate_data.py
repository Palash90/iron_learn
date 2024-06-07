import json
import numpy as np
m = int(input("Enter number of data points: "))
n = int(input("Enter number of features: "))

w = np.random.randint(2.0, 5.0, (1, n))
x = np.random.randint(-5.0, 5.0, (m, n))
y = x @ w.T
noise = np.random.normal(1, 5, y.shape)
y = y + noise

sigmoid = 1/(1 + np.exp(-y))

sigmoid = np.where(sigmoid > 0.5, 1.0, 0.0)

x = np.reshape(x, (m * n))
y = np.reshape(y, (m))
w = np.reshape(w, (n))
sigmoid = np.reshape(sigmoid, (m))

data = {"linear":{"m": m, "n": n, "x": x.tolist(), "y": y.tolist(), "w": w.tolist()},
        "logistic":{"m": m, "n": n, "x": x.tolist(), "y": sigmoid.tolist(), "w": w.tolist()}}

with open("data.json", "w") as outfile:
    json.dump(data, outfile)