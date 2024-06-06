import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, w, b, *argv):
    """
    Computes the cost over all examples
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value 
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model
      *argv : unused, for compatibility with regularized version below
    Returns:
      total_cost : (scalar) cost 
    """

    m, n = X.shape
    
    ### START CODE HERE ###
    loss_sum = 0 

    # Loop over each training example
    for i in range(m): 

        # First calculate z_wb = w[0]*X[i][0]+...+w[n-1]*X[i][n-1]+b
        z_wb = 0 
        # Loop over each feature
        for j in range(n): 
            # Add the corresponding term to z_wb
            z_wb_ij = w[j]*X[i][j]# Your code here to calculate w[j] * X[i][j]
            z_wb += z_wb_ij # equivalent to z_wb = z_wb + z_wb_ij
        # Add the bias term to z_wb
        z_wb += b # equivalent to z_wb = z_wb + b

        f_wb = sigmoid(z_wb) # Your code here to calculate prediction f_wb for a training example
        loss =  -y[i] * np.log(f_wb) - (1 - y[i]) * np.log(1 - f_wb)# Your code here to calculate loss for a training example

        loss_sum += loss # equivalent to loss_sum = loss_sum + loss

    total_cost = (1 / m) * loss_sum  
    ### END CODE HERE ### 

    return total_cost

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
            z_wb += b

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

