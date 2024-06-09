import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import json

X_train = np.array([[0.5, 1.5], [1.0, 1.0], [1.5, 0.5], [3.0, 0.5], [2.0, 2.0], [1.0, 2.5]])
y_train = np.array([0,0,0,1,1,1])

f = open('data.json')
data = json.load(f)

X_train = data["logistic"]["x"]
X_train = np.reshape(X_train, (data["logistic"]["m"], data["logistic"]["n"]))
y_train = data["logistic"]["y"]


def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z
         
    """
          
    ### START CODE HERE ### 
    g = 1 / (1 + np.exp(-z))
    ### END SOLUTION ###  
    
    return g

def compute_gradient(X, y, w, b, *argv): 
          m, n = X.shape
          dj_dw = np.zeros(w.shape)
          dj_db = 0.

          ### START CODE HERE ### 
          for i in range(m):
            # Calculate f_wb (exactly how you did it in the compute_cost function above)
            z_wb = 0
            # Loop over each feature
            for j in range(n): 
                # Add the corresponding term to z_wb
                z_wb_ij = X[i, j] * w[j]
                z_wb += z_wb_ij

            # Add bias term 
            z_wb += 0.0

            # Calculate the prediction from the model
            f_wb = sigmoid(z_wb)
            # Calculate the  gradient for b from this example
            dj_db_i = f_wb - y[i]# Your code here to calculate the error
            # add that to dj_db
            dj_db += dj_db_i
            
            # get dj_dw for each attribute
            for j in range(n):
                # You code here to calculate the gradient from the i-th example for j-th attribute
                dj_dw_ij = (f_wb - y[i])* X[i][j] 
                dj_dw[j] += dj_dw_ij

          # divide dj_db and dj_dw by total number of examples
          dj_dw = dj_dw / m
          dj_db = dj_db / m
          ### END CODE HERE ###

          return dj_db, dj_dw

def gradient_descent(X, y, w_in, b_in, gradient_function, alpha, num_iters, lambda_): 
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X :    (ndarray Shape (m, n) data, m examples by n features
      y :    (ndarray Shape (m,))  target value 
      w_in : (ndarray Shape (n,))  Initial values of parameters of the model
      b_in : (scalar)              Initial value of parameter of the model
      cost_function :              function to compute cost
      gradient_function :          function to compute gradient
      alpha : (float)              Learning rate
      num_iters : (int)            number of iterations to run gradient descent
      lambda_ : (scalar, float)    regularization constant
      
    Returns:
      w : (ndarray Shape (n,)) Updated values of parameters of the model after
          running gradient descent
      b : (scalar)                Updated value of parameter of the model after
          running gradient descent
    """
    
    # number of training examples
    m = len(X)
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)   

        # Update Parameters using w, b, alpha and gradient
        w_in = w_in - alpha * dj_dw               
        b_in = b_in - alpha * dj_db              
       
    return w_in, b_in

np.random.seed(1)
initial_w = np.zeros(3)
initial_b = 0

print(X_train.shape, initial_w.shape)

# Some gradient descent settings
iterations = 100_000
alpha = 0.0015
w, b = gradient_descent(X_train ,y_train, initial_w, initial_b, compute_gradient, alpha, iterations, 0)

print(w, b)

