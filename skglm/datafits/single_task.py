import numpy as np
from numpy.linalg import norm
from numba import njit
from numba import float64

from skglm.datafits.base import BaseDatafit
from skglm.utils import spectral_norm2


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

    global_lipschitz : float
        Global Lipschitz constant. Equal to
        norm(X, ord=2) ** 2 / n_samples.

    Note
    ----
    The class is jit compiled at fit time using Numba compiler.
    This allows for faster computations.
    """

    def __init__(self):
        pass

    def get_spec(self):
        spec = (
            ('Xty', float64[:]),
            ('lipschitz', float64[:]),
            ('global_lipschitz', float64),
        )
        return spec

    def params_to_dict(self):
        return dict()

    def initialize(self, X, y):
        self.Xty = X.T @ y
        n_features = X.shape[1]
        self.global_lipschitz = norm(X, ord=2) ** 2 / len(y)
        self.lipschitz = np.zeros(n_features, dtype=X.dtype)
        for j in range(n_features):
            self.lipschitz[j] = (X[:, j] ** 2).sum() / len(y)

    def initialize_sparse(self, X_data, X_indptr, X_indices, y):
        n_features = len(X_indptr) - 1
        self.Xty = np.zeros(n_features, dtype=X_data.dtype)

        self.global_lipschitz = spectral_norm2(X_data, X_indptr, X_indices, len(y))
        self.global_lipschitz /= len(y)

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

    def intercept_update_step(self, y, Xw):
        return np.mean(Xw - y)


@njit
def sigmoid(x):
    """Vectorwise sigmoid."""
    out = 1 / (1 + np.exp(- x))
    return out


class Logistic(BaseDatafit):
    r"""Logistic datafit with labels in {-1, 1}.

    The datafit reads::

    (1 / n_samples) * \sum_i log(1 + exp(-y_i * Xw_i))

    Attributes
    ----------
    lipschitz : array, shape (n_features,)
        The coordinatewise gradient Lipschitz constants. Equal to
        norm(X, axis=0) ** 2 / (4 * n_samples).

    global_lipschitz : float
        Global Lipschitz constant. Equal to
        norm(X, ord=2) ** 2 / (4 * n_samples).

    Note
    ----
    The class is jit compiled at fit time using Numba compiler.
    This allows for faster computations.
    """

    def __init__(self):
        pass

    def get_spec(self):
        spec = (
            ('lipschitz', float64[:]),
            ('global_lipschitz', float64),
        )
        return spec

    def params_to_dict(self):
        return dict()

    def raw_grad(self, y, Xw):
        """Compute gradient of datafit w.r.t ``Xw``."""
        return -y / (1 + np.exp(y * Xw)) / len(y)

    def raw_hessian(self, y, Xw):
        """Compute Hessian of datafit w.r.t ``Xw``."""
        exp_minus_yXw = np.exp(-y * Xw)
        return exp_minus_yXw / (1 + exp_minus_yXw) ** 2 / len(y)

    def initialize(self, X, y):
        self.lipschitz = (X ** 2).sum(axis=0) / (len(y) * 4)
        self.global_lipschitz = norm(X, ord=2) ** 2 / (len(y) * 4)

    def initialize_sparse(self, X_data, X_indptr, X_indices, y):
        n_features = len(X_indptr) - 1

        self.global_lipschitz = spectral_norm2(X_data, X_indptr, X_indices, len(y))
        self.global_lipschitz /= (len(y) * 4)

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

    def intercept_update_step(self, y, Xw):
        return np.mean(- y * sigmoid(- y * Xw)) / 4


class QuadraticSVC(BaseDatafit):
    """A Quadratic SVC datafit used for classification tasks.

    The datafit reads::

    1 / 2 * ||(y X).T w||^2_2

    Attributes
    ----------
    lipschitz : array, shape (n_features,)
        The coordinatewise gradient Lipschitz constants.
        Equal to norm(yXT, axis=0) ** 2.

    global_lipschitz : float
        Global Lipschitz constant. Equal to
        norm(yXT, ord=2) ** 2.

    Note
    ----
    The class is jit compiled at fit time using Numba compiler.
    This allows for faster computations.
    """

    def __init__(self):
        pass

    def get_spec(self):
        spec = (
            ('lipschitz', float64[:]),
            ('global_lipschitz', float64),
        )
        return spec

    def params_to_dict(self):
        return dict()

    def initialize(self, yXT, y):
        n_features = yXT.shape[1]
        self.lipschitz = np.zeros(n_features, dtype=yXT.dtype)
        self.global_lipschitz = norm(yXT, ord=2) ** 2
        for j in range(n_features):
            self.lipschitz[j] = norm(yXT[:, j]) ** 2

    def initialize_sparse(self, yXT_data, yXT_indptr, yXT_indices, y):
        n_features = len(yXT_indptr) - 1

        self.global_lipschitz = spectral_norm2(
            yXT_data, yXT_indptr, yXT_indices, len(y))

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


class Huber(BaseDatafit):
    """Huber datafit.

    The datafit reads::

    (1 / n_samples) * sum_{i=1}^{n_samples} f(y_i - Xw_i)

    where f is the Huber function:

    f(x) =
    1 / 2 * x^2                      if x <= delta
    delta * |x| - 1/2 * delta^2      if x > delta

    Attributes
    ----------
    delta : float
        Threshold hyperparameter.

    lipschitz : array, shape (n_features,)
        The coordinatewise gradient Lipschitz constants. Equal to
        norm(X, axis=0) ** 2 / n_samples.

    global_lipschitz : float
        Global Lipschitz constant. Equal to
        norm(X, ord=2) ** 2 / n_samples.

    Note
    ----
    The class is jit compiled at fit time using Numba compiler.
    This allows for faster computations.
    """

    def __init__(self, delta):
        self.delta = delta

    def get_spec(self):
        spec = (
            ('delta', float64),
            ('lipschitz', float64[:]),
            ('global_lipschitz', float64),
        )
        return spec

    def params_to_dict(self):
        return dict(delta=self.delta)

    def initialize(self, X, y):
        n_features = X.shape[1]
        self.lipschitz = np.zeros(n_features, dtype=X.dtype)
        self.global_lipschitz = 0.
        for j in range(n_features):
            self.lipschitz[j] = (X[:, j] ** 2).sum() / len(y)
            self.global_lipschitz += (X[:, j] ** 2).sum() / len(y)

    def initialize_sparse(self, X_data, X_indptr, X_indices, y):
        n_features = len(X_indptr) - 1

        self.global_lipschitz = spectral_norm2(X_data, X_indptr, X_indices, len(y))
        self.global_lipschitz /= len(y)

        self.lipschitz = np.zeros(n_features, dtype=X_data.dtype)
        for j in range(n_features):
            nrm2 = 0.
            for idx in range(X_indptr[j], X_indptr[j + 1]):
                nrm2 += X_data[idx] ** 2
            self.lipschitz[j] = nrm2 / len(y)

    def value(self, y, w, Xw):
        n_samples = len(y)
        res = 0.
        for i in range(n_samples):
            residual = abs(y[i] - Xw[i])
            if residual < self.delta:
                res += 0.5 * residual ** 2
            else:
                res += self.delta * residual - 0.5 * self.delta ** 2
        return res / n_samples

    def gradient_scalar(self, X, y, w, Xw, j):
        n_samples = len(y)
        grad_j = 0.
        for i in range(n_samples):
            residual = y[i] - Xw[i]
            if abs(residual) < self.delta:
                grad_j += - X[i, j] * residual
            else:
                grad_j += - X[i, j] * np.sign(residual) * self.delta
        return grad_j / n_samples

    def gradient_scalar_sparse(self, X_data, X_indptr, X_indices, y, Xw, j):
        grad_j = 0.
        for i in range(X_indptr[j], X_indptr[j + 1]):
            residual = y[X_indices[i]] - Xw[X_indices[i]]
            if np.abs(residual) < self.delta:
                grad_j += - X_data[i] * residual
            else:
                grad_j += - X_data[i] * np.sign(residual) * self.delta
        return grad_j / len(Xw)

    def full_grad_sparse(
            self, X_data, X_indptr, X_indices, y, Xw):
        n_features = X_indptr.shape[0] - 1
        n_samples = y.shape[0]
        grad = np.zeros(n_features, dtype=Xw.dtype)
        for j in range(n_features):
            grad_j = 0.
            for i in range(X_indptr[j], X_indptr[j + 1]):
                residual = y[X_indices[i]] - Xw[X_indices[i]]
                if np.abs(residual) < self.delta:
                    grad_j += - X_data[i] * residual
                else:
                    grad_j += - X_data[i] * np.sign(residual) * self.delta
            grad[j] = grad_j / n_samples
        return grad

    def intercept_update_step(self, y, Xw):
        n_samples = len(y)
        update = 0.
        for i in range(n_samples):
            residual = y[i] - Xw[i]
            if abs(residual) < self.delta:
                update -= residual
            else:
                update -= np.sign(residual) * self.delta
        return update / n_samples


class Poisson(BaseDatafit):
    r"""Poisson datafit.

    The datafit reads::

    (1 / n_samples) * \sum_i (exp(Xw_i) - y_i * Xw_i)

    Note:
    ----
    The class is jit compiled at fit time using Numba compiler.
    This allows for faster computations.
    """

    def __init__(self):
        pass

    def get_spec(self):
        pass

    def params_to_dict(self):
        return dict()

    def initialize(self, X, y):
        pass

    def initialize_sparse(self, X_data, X_indptr, X_indices, y):
        pass

    def raw_grad(self, y, Xw):
        """Compute gradient of datafit w.r.t ``Xw``."""
        return (np.exp(Xw) - y) / len(y)

    def raw_hessian(self, y, Xw):
        """Compute Hessian of datafit w.r.t ``Xw``."""
        return np.exp(Xw) / len(y)

    def value(self, y, w, Xw):
        return np.sum(np.exp(Xw) - y * Xw) / len(y)

    def gradient_scalar(self, X, y, w, Xw, j):
        return (X[:, j] @ (np.exp(Xw) - y)) / len(y)

    def full_grad_sparse(self, X_data, X_indptr, X_indices, y, Xw):
        n_features = X_indptr.shape[0] - 1
        grad = np.zeros(n_features, dtype=X_data.dtype)
        for j in range(n_features):
            grad[j] = 0.
            for i in range(X_indptr[j], X_indptr[j + 1]):
                grad[j] += X_data[i] * (
                    np.exp(Xw[X_indices[i]] - y[X_indices[i]])) / len(y)
        return grad

    def gradient_scalar_sparse(self, X_data, X_indptr, X_indices, y, Xw, j):
        grad = 0.
        for i in range(X_indptr[j], X_indptr[j + 1]):
            idx_i = X_indices[i]
            grad += X_data[i] * (np.exp(Xw[idx_i]) - y[idx_i])
        return grad / len(y)

    def intercept_update_self(self, y, Xw):
        pass
