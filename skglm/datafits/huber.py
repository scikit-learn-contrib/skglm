import numpy as np
from numba import float64
from skglm.datafits.base import BaseDatafit, jit_factory


spec_huber = [
    ('delta', float64),
    ('Xty', float64[:]),
    ('lipschitz', float64[:]),
]


class Huber(BaseDatafit):
    """Huber datafit.

    The datafit reads::

    1 / n_samples * sum (f_k(y_k - (X w)_k))

    f_k(x) = 1 / 2 * x^2 si x <= sigma
    f_k(x) = sigma * | x | - 1 /2 * sigma si x > sigma


    Attributes
    ----------
    Xty : array, shape (n_features,)
        Pre-computed quantity used during the gradient evaluation. Equal to X.T @ y.

    lipschitz : array, shape (n_features,)
        The coordinatewise gradient Lipschitz constants.

    Note
    ----
    The class Quadratic is subsequently decorated with a @jitclass decorator with
    the `jit_factory` function to be compiled. This allows for faster computations
    using Numba JIT compiler.
    """

    def __init__(self, delta):
        self.delta = delta

    def initialize(self, X, y):
        self.Xty = X.T @ y
        n_features = X.shape[1]
        self.lipschitz = np.zeros(n_features, dtype=X.dtype)
        for j in range(n_features):
            self.lipschitz[j] = (np.where(np.abs(y) < self.delta,
                                          X[:, j] ** 2, 0)).sum() / len(y)

    def initialize_sparse(
            self, X_data, X_indptr, X_indices, y):
        n_features = len(X_indptr) - 1
        self.Xty = np.zeros(n_features, dtype=X_data.dtype)
        self.lipschitz = np.zeros(n_features, dtype=X_data.dtype)
        for j in range(n_features):
            nrm2 = 0.
            xty = 0
            for idx in range(X_indptr[j], X_indptr[j + 1]):
                if np.abs(y[idx]) < self.delta:
                    nrm2 += X_data[idx] ** 2
                xty += X_data[idx] * y[X_indices[idx]]

            self.lipschitz[j] = nrm2 / len(y)
            self.Xty[j] = xty

    def value(self, y, w, Xw):
        norm_1 = np.abs(y - Xw)
        loss = np.where(norm_1 < self.delta,
                        0.5 * norm_1 ** 2,
                        self.delta * norm_1 - 0.5 * self.delta ** 2)
        return np.sum(loss) / len(Xw)

    def gradient_scalar(self, X, y, w, Xw, j):
        R = y - Xw
        if np.abs(R) < self.delta:
            return (X[:, j] @ Xw - self.Xty[j]) / len(Xw)
        else:
            return X[:, j] @ np.sign(-R) * self.delta / len(Xw)

    def gradient_scalar_sparse(self, X_data, X_indptr, X_indices, y, Xw, j):
        grad_j = 0.
        for i in range(X_indptr[j], X_indptr[j + 1]):
            R = y[X_indices[i]] - Xw[X_indices[i]]
            if np.abs(R) < self.delta:
                grad_j += X_data[i] * Xw[X_indices[i]] - self.Xty[j]
            else:
                grad_j += X_data[i] * np.sign(R) * self.delta
        return grad_j / len(Xw)

    def full_grad_sparse(
            self, X_data, X_indptr, X_indices, y, Xw):
        n_features = X_indptr.shape[0] - 1
        n_samples = y.shape[0]
        grad = np.zeros(n_features, dtype=Xw.dtype)
        for j in range(n_features):
            grad_j = 0.
            for i in range(X_indptr[j], X_indptr[j + 1]):
                R = y[X_indices[i]] - Xw[X_indices[i]]
                if np.abs(R) < self.delta:
                    grad_j += X_data[i] * Xw[X_indices[i]] - self.Xty[j]
                else:
                    grad_j += X_data[i] * np.sign(R) * self.delta
            grad[j] = grad_j / n_samples
        return grad

Huber, Huber_32 = jit_factory(Huber, spec_huber)

if __name__ == '__main__':
    from skglm import GeneralizedLinearEstimator
    from skglm.penalties import L1

    clf = GeneralizedLinearEstimator(
        Huber(0.5), L1(0.1), is_classif=False
    )
    X = np.random.randn(10, 10)
    y = np.random.randn(10)
    #import ipdb; ipdb.set_trace()
    clf.fit(X, y)
