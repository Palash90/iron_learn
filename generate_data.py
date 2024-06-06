import json
import numpy as np
m = int(input("Enter number of data points: "))
n = int(input("Enter number of features: "))

w = np.random.randint(2, 10, (1, n))
x = np.random.randint(5, 10, (m, n))
y = x @ w.T

x = np.reshape(x, (m * n))
y = np.reshape(y, (m))
w = np.reshape(w, (n))

data = {"m": m, "n": n, "x": x.tolist(), "y": y.tolist(), "w": w.tolist()}

with open("data.json", "w") as outfile:
    json.dump(data, outfile)