

import numpy as np

def random_forest_vote(predictions):
    predictions = np.array(predictions)
    result = []

    for col in predictions.T:
        values, counts = np.unique(col, return_counts=True)
        result.append(int(values[np.argmax(counts)]))

    return result