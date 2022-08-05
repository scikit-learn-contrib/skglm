import numpy as np


def compute_alpha_max(X, y, is_sparse=False):
    n_samples = len(y)

    if is_sparse:
        alpha_max = 0.

        for j in range(X.shape[1]):
            tmp = 0.
            for i in range(X.indptr[j], X.indptr[j+1]):
                tmp += X.data[i] * y[X.indices[i]]
            tmp /= 2 * n_samples
            alpha_max = max(alpha_max, np.abs(tmp))

        return alpha_max
    else:
        return np.linalg.norm(X.T @ y, ord=np.inf) / (2 * n_samples)
