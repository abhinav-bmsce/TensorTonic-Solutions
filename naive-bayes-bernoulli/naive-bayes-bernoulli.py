import numpy as np

def naive_bayes_bernoulli(X_train, y_train, X_test):
    """
    Compute log-likelihood P(y|x) for Bernoulli Naive Bayes.
    """
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)

    classes = np.sort(np.unique(y_train))
    n_classes = len(classes)
    n_test = X_test.shape[0]
    n_features = X_train.shape[1]

    log_posteriors = np.zeros((n_test, n_classes))

    for idx, c in enumerate(classes):
        X_c = X_train[y_train == c]
        n_c = len(X_c)

        # Log prior
        log_prior = np.log(n_c / len(y_train))

        # Bernoulli likelihood with Laplace smoothing (alpha=1)
        theta = (np.sum(X_c, axis=0) + 1) / (n_c + 2)

        # Log posterior for all test samples
        log_likelihood = (
            X_test * np.log(theta) +
            (1 - X_test) * np.log(1 - theta)
        ).sum(axis=1)

        log_posteriors[:, idx] = log_prior + log_likelihood

    return log_posteriors