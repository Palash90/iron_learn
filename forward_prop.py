import numpy as np
# Define the activation function
def g(z):
    return 1/( 1 +  np.exp(-z))

def my_dense(a_in, W, b):
    """
    Computes dense layer
    Args:
      a_in (ndarray (n, )) : Data, 1 example 
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (j, )) : bias vector, j units  
    Returns
      a_out (ndarray (j,))  : j units|
    """
    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):               
        w = W[:,j]                                    
        z = np.dot(w, a_in) + b[j]         
        print("Z ", z)
        gz = g(z)
        print("g", gz)
        a_out = gz               
    return(a_out)

# A 2 layer Neural Network
def my_sequential(x, W1, b1, W2, b2):
    a1 = my_dense(x,  W1, b1)
    a2 = my_dense(a1, W2, b2)
    return(a2)

# Prediction
def my_predict(X, W1, b1, W2, b2):
    m = X.shape[0]
    p = np.zeros((m,1))
    for i in range(m):
        p[i,0] = my_sequential(X[i], W1, b1, W2, b2)
    return(p)

X = np.array([1,2])
W1 = np.array([[3, 4],[6, 7]])
b1 = np.array([8, 9])
W2 = np.array([[7]])
b2 = np.array([8])
my_predict(X, W1, b1, W2, b2)
