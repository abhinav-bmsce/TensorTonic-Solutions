


import numpy as np

def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    Formula: c_j = (1 / |S_j|) * sum_{p in S_j} p
    
    Arguments:
    points: array-like, shape (N, D) - Coordinates of N data points
    assignments: array-like, shape (N,) - Cluster index assignment for each point
    k: int - Total number of clusters
    
    Returns:
    A nested list of shape (k, D) containing the updated centroid positions
    sorted by cluster index from 0 to k-1.
    """
    points = np.array(points)
    assignments = np.array(assignments)
    
    # Get the feature dimension D
    d = points.shape[1]
    
    # Initialize an empty array for the new centroids
    new_centroids = np.zeros((k, d))
    
    for j in range(k):
        # Create a mask for points belonging to cluster j
        cluster_mask = (assignments == j)
        
        # Extract points belonging to cluster j
        cluster_points = points[cluster_mask]
        
        # If the cluster has assigned points, compute the mean along axis 0
        if len(cluster_points) > 0:
            new_centroids[j] = np.mean(cluster_points, axis=0)
        else:
            # Fallback handling if a cluster becomes empty
            new_centroids[j] = np.zeros(d)
            
    # Convert the final 2D NumPy array to a native Python nested list
    return new_centroids.tolist()