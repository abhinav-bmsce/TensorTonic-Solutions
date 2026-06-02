import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    y = np.array(y)

    # Handle empty node
    if len(y) == 0:
        return 0.0

    # Class probabilities
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)

    # Stable entropy computation
    entropy = -np.sum(probs * np.log2(probs))

    return float(entropy)