




import numpy as np

def linear_regression_closed_form(X, y):
    """
    Compute the optimal weight vector using the normal equation.
    Formula: w = (X^T * X)^-1 * X^T * y
    
    Arguments:
    X: array-like, shape (n_samples, d_features) - Feature matrix
    y: array-like, shape (n_samples,) or (n_samples, 1) - Target vector
    
    Returns:
    w: 1D or 2D numpy array containing the d-dimensional weight vector
    """
    # Convert inputs to numpy arrays for matrix operations
    X = np.array(X)
    y = np.array(y)
    
    # 1. Compute X transpose: X^T
    X_transpose = X.T
    
    # 2. Compute the dot product of X^T and X
    XTX = np.dot(X_transpose, X)
    
    # 3. Compute the inverse of (X^T * X)
    # Using np.linalg.inv to solve for the analytical inverse
    XTX_inv = np.linalg.inv(XTX)
    
    # 4. Compute the dot product of X^T and y
    XTy = np.dot(X_transpose, y)
    
    # 5. Compute the final weights: w = (X^T * X)^-1 * X^T * y
    w = np.dot(XTX_inv, XTy)
    
    return w