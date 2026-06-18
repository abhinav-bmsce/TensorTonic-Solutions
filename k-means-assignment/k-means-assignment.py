



import numpy as np

def k_means_assignment(points, centroids):
    """
    Assign each point to the nearest centroid.
    Formula: assignment(p) = argmin_j sum_{d=1}^D (p_d - c_{j,d})^2
    
    Arguments:
    points: array-like, shape (N, D) - Coordinates of N data points
    centroids: array-like, shape (K, D) - Coordinates of K centroids
    
    Returns:
    A flat Python list of length N containing the index of the nearest centroid 
    for each data point.
    """
    # Convert inputs to numpy arrays
    points = np.array(points)        # Shape: (N, D)
    centroids = np.array(centroids)  # Shape: (K, D)
    
    # Expand dimensions to leverage numpy broadcasting:
    # points[:, np.newaxis, :] expands points to shape (N, 1, D)
    differences = points[:, np.newaxis, :] - centroids
    
    # Compute squared Euclidean distance across the feature dimension (axis=2)
    squared_distances = np.sum(differences ** 2, axis=2)
    
    # For each data point, find the index of the minimum distance (axis=1)
    assignments = np.argmin(squared_distances, axis=1)
    
    # Convert the resulting numpy array to a native Python list
    return assignments.tolist()