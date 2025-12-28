import numpy as np


np.random.seed(1610612741)

print("Five random numbers from NumPY on CPU:")
for i in range(5):
    rand = np.random.standard_normal()
    print(rand)