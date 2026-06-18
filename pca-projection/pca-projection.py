

import numpy as np

def pca_projection(X, k):
    X = np.array(X, dtype=float)

    X_centered = X - np.mean(X, axis=0)

    C = np.cov(X_centered, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(C)

    idx = np.argsort(eigenvalues)[::-1]

    W = eigenvectors[:, idx[:k]]

    X_proj = X_centered @ W

    return X_proj.tolist()