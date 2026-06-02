import numpy as np

def decision_tree_split(X, y):
    """
    Find the best feature and threshold to split the data.
    """
    X = np.array(X)
    y = np.array(y)

    def gini(labels):
        if len(labels) == 0:
            return 0.0
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / len(labels)
        return 1.0 - np.sum(probs ** 2)

    n_samples, n_features = X.shape
    parent_gini = gini(y)

    best_gain = -1
    best_feature = 0
    best_threshold = 0

    for feature in range(n_features):
        values = np.unique(X[:, feature])

        for i in range(len(values) - 1):
            threshold = (values[i] + values[i + 1]) / 2.0

            left_mask = X[:, feature] <= threshold
            right_mask = X[:, feature] > threshold

            y_left = y[left_mask]
            y_right = y[right_mask]

            weighted_gini = (
                len(y_left) / n_samples * gini(y_left)
                + len(y_right) / n_samples * gini(y_right)
            )

            gain = parent_gini - weighted_gini

            if (gain > best_gain or
                (gain == best_gain and feature < best_feature) or
                (gain == best_gain and feature == best_feature and threshold < best_threshold)):
                best_gain = gain
                best_feature = feature
                best_threshold = threshold

    return [best_feature, float(best_threshold)]