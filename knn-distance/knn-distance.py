import numpy as np

def knn_distance(X_train, X_test, k):
    """
    Compute pairwise distances and return k nearest neighbor indices.
    """
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    # Handle 1D inputs
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    # Pairwise Euclidean distances
    distances = np.sqrt(
        np.sum((X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]) ** 2, axis=2)
    )

    n_train = X_train.shape[0]
    k_actual = min(k, n_train)

    # Get indices of nearest neighbors
    neighbors = np.argsort(distances, axis=1)[:, :k_actual]

    # Pad with -1 if k > n_train
    if k > n_train:
        padding = np.full((X_test.shape[0], k - n_train), -1, dtype=int)
        neighbors = np.hstack((neighbors, padding))

    return neighbors.astype(int)