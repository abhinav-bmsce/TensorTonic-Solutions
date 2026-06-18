

import numpy as np

def _entropy(y):
    """
    Helper: Compute Shannon entropy (base 2) for labels y.
    """
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    vals, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0

def information_gain(y, split_mask):
    """
    Compute Information Gain of a binary split on labels y.
    Use the _entropy() helper above.
    """
    # Ensure inputs are numpy arrays for clean boolean indexing
    y = np.asarray(y)
    split_mask = np.asarray(split_mask, dtype=bool)
    
    # Total number of samples in the parent node
    N = y.size
    if N == 0:
        return 0.0
        
    # 1. Compute Parent Entropy: H(Y)
    H_parent = _entropy(y)
    
    # 2. Split the labels into Left and Right child nodes using the boolean mask
    y_L = y[split_mask]
    y_R = y[~split_mask]
    
    # Track the number of samples in each child node
    n_L = y_L.size
    n_R = y_R.size
    
    # 3. Compute the Entropies of the children: H(Y_L) and H(Y_R)
    H_L = _entropy(y_L)
    H_R = _entropy(y_R)
    
    # 4. Calculate the weighted conditional entropy of the children
    # Weighted H = (n_L / N) * H(Y_L) + (n_R / N) * H(Y_R)
    weighted_entropy_children = (n_L / N) * H_L + (n_R / N) * H_R
    
    # 5. Information Gain = H(Parent) - Weighted Children Entropy
    ig = H_parent - weighted_entropy_children
    
    return float(ig)