import numpy as np

def estimate_gaussian(X): 
    """
    Calculates mean and variance of all features 
    in the dataset
    
    Args:
        X (ndarray): (m, n) Data matrix
    
    Returns:
        mu (ndarray): (n,) Mean of all features
        var (ndarray): (n,) Variance of all features
    """

    m, n = X.shape
    
    ### START CODE HERE ### 
    mu = 1 / m * np.sum(X, axis = 0)
    var = 1 / m * np.sum((X - mu) ** 2, axis = 0)
    
    ### END CODE HERE ### 
        
    return mu, var

def select_threshold(y_val, p_val): 
    """
    Finds the best threshold to use for selecting outliers 
    based on the results from a validation set (p_val) 
    and the ground truth (y_val)
    
    Args:
        y_val (ndarray): Ground truth on validation set
        p_val (ndarray): Results on validation set
        
    Returns:
        epsilon (float): Threshold chosen 
        F1 (float):      F1 score by choosing epsilon as threshold
    """ 

    best_epsilon = 0
    best_F1 = 0
    F1 = 0
    
    step_size = (max(p_val) - min(p_val)) / 1000
    
    for epsilon in np.arange(min(p_val), max(p_val), step_size):
    
        ### START CODE HERE ### 
        predictions = (p_val < epsilon)# Your code here to calculate predictions for each example using epsilon as threshold

        tp = np.sum((predictions == 1) & (y_val == 1))# Your code here to calculate number of true positives
        fp = sum((predictions == 1) & (y_val == 0))# Your code here to calculate number of false positives
        fn = np.sum((predictions == 0) & (y_val == 1))# Your code here to calculate number of false negatives

        prec = tp / (tp + fp) # Your code here to calculate precision
        rec = tp / (tp + fn)# Your code here to calculate recall

        F1 = 2 * prec * rec / (prec + rec)# Your code here to calculate F1
        ### END CODE HERE ### 

        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon
        
    return best_epsilon, best_F1

