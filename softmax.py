def my_softmax(z):  
    """ Softmax converts a vector of values to a probability distribution.
    Args:
      z (ndarray (N,))  : input data, N features
    Returns:
      a (ndarray (N,))  : softmax of z
    """    
    ### START CODE HERE ### 
    N = len(z)
    a= np.zeros(N)
    ez_sum = 0
    for k in range(N):         # loop over number of outputs             
        ez_sum += np.exp(z[k]) # sum exp(z[k]) to build the shared denominator      
    for j in range(N):      # loop over number of outputs again                
        a[j] = np.exp(z[j])/ez_sum             # divide each the exp of each output by the denominator   
    return(a)