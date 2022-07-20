import numpy as np
from numpy.linalg import norm
from numba import njit
from numba import float64

from skglm.datafits.base import BaseDatafit, jit_factory


spec_quadratic = [
    ('Xty', float64[:]),
    ('lipschitz', float64[:]),
]


class Quadratic(BaseDatafit):
    """Quadratic datafit.

    The datafit reads::

    (1 / (2 * n_samples)) * ||y - X w||^2_2

    Attributes
    ----------
    Xty : array, shape (n_features,)
        Pre-computed quantity used during the gradient evaluation. Equal to X.T @ y.

    lipschitz : array, shape (n_features,)
        The coordinatewise gradient Lipschitz constants. Equal to
        norm(X, axis=0) ** 2 / n_samples.

    Note
    ----
    The class Quadratic is subsequently decorated with a @jitclass decorator with
    the `jit_factory` function to be compiled. This allows for faster computations
    using Numba JIT compiler.
    """

    def __init__(self):
        pass

    def initialize(self, X, y):
        self.Xty = X.T @ y
        n_features = X.shape[1]
        self.lipschitz = np.zeros(n_features, dtype=X.dtype)
        for j in range(n_features):
            self.lipschitz[j] = (X[:, j] ** 2).sum() / len(y)

    def initialize_sparse(
            self, X_data, X_indptr, X_indices, y):
        n_features = len(X_indptr) - 1
        self.Xty = np.zeros(n_features, dtype=X_data.dtype)
        self.lipschitz = np.zeros(n_features, dtype=X_data.dtype)
        for j in range(n_features):
            nrm2 = 0.
            xty = 0
            for idx in range(X_indptr[j], X_indptr[j + 1]):
                nrm2 += X_data[idx] ** 2
                xty += X_data[idx] * y[X_indices[idx]]

            self.lipschitz[j] = nrm2 / len(y)
            self.Xty[j] = xty

    def value(self, y, w, Xw):
        return np.sum((y - Xw) ** 2) / (2 * len(Xw))

    def gradient_scalar(self, X, y, w, Xw, j):
        return (X[:, j] @ Xw - self.Xty[j]) / len(Xw)

    def gradient_scalar_sparse(self, X_data, X_indptr, X_indices, y, Xw, j):
        XjTXw = 0.
        for i in range(X_indptr[j], X_indptr[j+1]):
            XjTXw += X_data[i] * Xw[X_indices[i]]
        return (XjTXw - self.Xty[j]) / len(Xw)

    def full_grad_sparse(
            self, X_data, X_indptr, X_indices, y, Xw):
        n_features = X_indptr.shape[0] - 1
        n_samples = y.shape[0]
        grad = np.zeros(n_features, dtype=Xw.dtype)
        for j in range(n_features):
            XjTXw = 0.
            for i in range(X_indptr[j], X_indptr[j + 1]):
                XjTXw += X_data[i] * Xw[X_indices[i]]
            grad[j] = (XjTXw - self.Xty[j]) / n_samples
        return grad

    def update_intercept(self, y, Xw):
        return np.sum(Xw - y) / len(Xw)


Quadratic, Quadratic_32 = jit_factory(Quadratic, spec_quadratic)


@njit
def sigmoid(x):
    """Vectorwise sigmoid."""
    out = 1 / (1 + np.exp(- x))
    return out


spec_logistic = [
    ('lipschitz', float64[:]),
]


class _Logistic(BaseDatafit):
    r"""Logistic datafit with labels in {-1, 1}.

    The datafit reads::

    (1 / n_samples) * \sum_i log(1 + exp(-y_i * Xw_i))

    Attributes
    ----------
    lipschitz : array, shape (n_features,)
        The coordinatewise gradient Lipschitz constants. Equal to
        norm(X, axis=0) ** 2 / (4 * n_samples).

    Note
    ----
    The class _Logistic is subsequently decorated with a @jitclass decorator with
    the `jit_factory` function to be compiled. This allows for faster computations
    using Numba JIT compiler.
    """

    def __init__(self):
        pass

    def initialize(self, X, y):
        self.lipschitz = (X ** 2).sum(axis=0) / (len(y) * 4)

    def initialize_sparse(self, X_data, X_indptr, X_indices, y):
        n_features = len(X_indptr) - 1
        self.lipschitz = np.zeros(n_features, dtype=X_data.dtype)
        for j in range(n_features):
            Xj = X_data[X_indptr[j]:X_indptr[j+1]]
            self.lipschitz[j] = (Xj ** 2).sum() / (len(y) * 4)

    def value(self, y, w, Xw):
        return np.log(1. + np.exp(- y * Xw)).sum() / len(y)

    def gradient_scalar(self, X, y, w, Xw, j):
        return (- X[:, j] @ (y * sigmoid(- y * Xw))) / len(y)

    def full_grad_sparse(
            self, X_data, X_indptr, X_indices, y, Xw):
        n_features = X_indptr.shape[0] - 1
        grad = np.zeros(n_features, dtype=X_data.dtype)
        for j in range(n_features):
            grad[j] = 0.
            for i in range(X_indptr[j], X_indptr[j + 1]):
                grad[j] -= X_data[i] * y[X_indices[i]] * sigmoid(
                    - y[X_indices[i]] * Xw[X_indices[i]]) / len(y)
        return grad

    def gradient_scalar_sparse(self, X_data, X_indptr, X_indices, y, Xw, j):
        grad = 0.
        for i in range(X_indptr[j], X_indptr[j+1]):
            idx_i = X_indices[i]
            grad -= X_data[i] * y[idx_i] * sigmoid(- y[idx_i] * Xw[idx_i])
        return grad / len(Xw)

    def update_intercept(self, y, Xw):
        return - np.sum((y * sigmoid(- y * Xw))) / (4 * len(Xw))


Logistic, Logistic_32 = jit_factory(_Logistic, spec_logistic)


spec_quadratic_svc = [
    ('lipschitz', float64[:]),
]


class _QuadraticSVC(BaseDatafit):
    """A Quadratic SVC datafit used for classification tasks.

    The datafit reads::

    1 / 2 * ||(y X).T w||^2_2

    Attributes
    ----------
    lipschitz : array, shape (n_features,)
        The coordinatewise gradient Lipschitz constants.

    Note
    ----
    The class _Logistic is subsequently decorated with a @jitclass decorator with
    the `jit_factory` function to be compiled. This allows for faster computations
    using Numba JIT compiler.
    """

    def __init__(self):
        pass

    def initialize(self, yXT, y):
        n_features = yXT.shape[1]
        self.lipschitz = np.zeros(n_features, dtype=yXT.dtype)
        for j in range(n_features):
            self.lipschitz[j] = norm(yXT[:, j]) ** 2

    def initialize_sparse(
            self, yXT_data, yXT_indptr, yXT_indices, y):
        n_features = len(yXT_indptr) - 1
        self.lipschitz = np.zeros(n_features, dtype=yXT_data.dtype)
        for j in range(n_features):
            nrm2 = 0.
            for idx in range(yXT_indptr[j], yXT_indptr[j + 1]):
                nrm2 += yXT_data[idx] ** 2
            self.lipschitz[j] = nrm2

    def value(self, y, w, yXTw):
        return (yXTw ** 2).sum() / 2 - np.sum(w)

    def gradient_scalar(self, yXT, y, w, yXTw, j):
        return yXT[:, j].T @ yXTw - 1.0

    def gradient_scalar_sparse(
            self, yXT_data, yXT_indptr, yXT_indices, y, yXTw, j):
        # Compute grad[j] = yXT[:, j].T @ yXTw
        yXjyXTw = 0.
        for i in range(yXT_indptr[j], yXT_indptr[j+1]):
            yXjyXTw += yXT_data[i] * yXTw[yXT_indices[i]]
        return yXjyXTw - 1

    def full_grad_sparse(
            self, yXT_data, yXT_indptr, yXT_indices, y, yXTw):
        n_features = yXT_indptr.shape[0] - 1
        grad = np.zeros(n_features, dtype=yXT_data.dtype)
        for j in range(n_features):
            # Compute grad[j] = yXT[:, j].T @ yXTw
            yXjyXTw = 0.
            for i in range(yXT_indptr[j], yXT_indptr[j + 1]):
                yXjyXTw += yXT_data[i] * yXTw[yXT_indices[i]]
            grad[j] = yXjyXTw - 1
        return grad


QuadraticSVC, QuadraticSVC_32 = jit_factory(_QuadraticSVC, spec_quadratic_svc)
