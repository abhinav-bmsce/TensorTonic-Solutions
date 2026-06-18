


import numpy as np

def majority_classifier(y_train, X_test):
    if len(X_test) == 0:
        return np.array([], dtype=int)

    values, counts = np.unique(y_train, return_counts=True)
    majority = values[np.argmax(counts)]

    return np.full(len(X_test), majority, dtype=int)