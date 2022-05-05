import numpy as np
from numba import float64
from numba.experimental import jitclass
from skglm.datafits import BaseDatafit


spec_huber = [
    ('delta', float64),
    ('lipschitz', float64[:])
]


@jitclass(spec_huber)
class Huber(BaseDatafit):
    """Huber datafit.
    The datafit reads::

    1 / n_samples * sum (f(y_k - (X w)_k))

    f(x) = 1 / 2 * x^2 if x <= delta
    f(x) = delta * | x | - 1 /2 * delta^2 if x > delta


    Attributes
    ----------
    lipschitz : array, shape (n_features,)
        The coordinatewise gradient Lipschitz constants.
    """

    def __init__(self, delta):
        self.delta = delta

    def initialize(self, X, y):
        n_features = X.shape[1]
        self.lipschitz = np.zeros(n_features, dtype=X.dtype)
        for j in range(n_features):
            self.lipschitz[j] = (X[:, j] ** 2).sum() / len(y)

    def initialize_sparse(
            self, X_data, X_indptr, X_indices, y):
        n_features = len(X_indptr) - 1
        self.lipschitz = np.zeros(n_features, dtype=X_data.dtype)
        for j in range(n_features):
            nrm2 = 0.
            for idx in range(X_indptr[j], X_indptr[j + 1]):
                nrm2 += X_data[idx] ** 2
            self.lipschitz[j] = nrm2 / len(y)

    def value(self, y, w, Xw):
        norm_1 = np.abs(y - Xw)
        loss = np.where(norm_1 < self.delta,
                        0.5 * norm_1 ** 2,
                        self.delta * norm_1 - 0.5 * self.delta ** 2)
        return np.sum(loss) / len(Xw)

    def gradient_scalar(self, X, y, w, Xw, j):
        R = y - Xw
        return - X[:, j] @ np.where(np.abs(R) < self.delta,
                                    R,
                                    np.sign(R) * self.delta) / len(Xw)

    def gradient_scalar_sparse(self, X_data, X_indptr, X_indices, y, Xw, j):
        grad_j = 0.
        for i in range(X_indptr[j], X_indptr[j + 1]):
            diff_indice_i = y[X_indices[i]] - Xw[X_indices[i]]
            if np.abs(diff_indice_i) < self.delta:
                grad_j += X_data[i] * (-diff_indice_i)
            else:
                grad_j += X_data[i] * np.sign(-diff_indice_i) * self.delta
        return grad_j / len(Xw)

    def full_grad_sparse(
            self, X_data, X_indptr, X_indices, y, Xw):
        n_features = X_indptr.shape[0] - 1
        n_samples = y.shape[0]
        grad = np.zeros(n_features, dtype=Xw.dtype)
        for j in range(n_features):
            grad_j = 0.
            for i in range(X_indptr[j], X_indptr[j + 1]):
                diff_indice_i = y[X_indices[i]] - Xw[X_indices[i]]
                if np.abs(diff_indice_i) < self.delta:
                    grad_j += X_data[i] * (-diff_indice_i)
                else:
                    grad_j += X_data[i] * np.sign(-diff_indice_i) * self.delta
            grad[j] = grad_j / n_samples
        return grad
