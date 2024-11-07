import numpy as np
from numpy.linalg import norm
from numba import njit
from numba import float64, int64, bool_

from skglm.datafits.base import BaseDatafit
from skglm.utils.sparse_ops import spectral_norm, _sparse_xj_dot


class Quadratic(BaseDatafit):
    """Quadratic datafit.

    The datafit reads:

    .. math:: 1 / (2 xx  n_"samples") ||y - Xw||_2 ^ 2

    Attributes
    ----------
    Xty : array, shape (n_features,)
        Pre-computed quantity used during the gradient evaluation.
        Equal to ``X.T @ y``.

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
        )
        return spec

    def params_to_dict(self):
        return dict()

    def get_lipschitz(self, X, y):
        n_features = X.shape[1]

        lipschitz = np.zeros(n_features, dtype=X.dtype)
        for j in range(n_features):
            lipschitz[j] = (X[:, j] ** 2).sum() / len(y)

        return lipschitz

    def get_lipschitz_sparse(self, X_data, X_indptr, X_indices, y):
        n_features = len(X_indptr) - 1
        lipschitz = np.zeros(n_features, dtype=X_data.dtype)

        for j in range(n_features):
            nrm2 = 0.
            for idx in range(X_indptr[j], X_indptr[j + 1]):
                nrm2 += X_data[idx] ** 2

            lipschitz[j] = nrm2 / len(y)

        return lipschitz

    def initialize(self, X, y):
        self.Xty = X.T @ y

    def initialize_sparse(self, X_data, X_indptr, X_indices, y):
        n_features = len(X_indptr) - 1
        self.Xty = np.zeros(n_features, dtype=X_data.dtype)

        for j in range(n_features):
            xty = 0
            for idx in range(X_indptr[j], X_indptr[j + 1]):
                xty += X_data[idx] * y[X_indices[idx]]

            self.Xty[j] = xty

    def get_global_lipschitz(self, X, y):
        return norm(X, ord=2) ** 2 / len(y)

    def get_global_lipschitz_sparse(self, X_data, X_indptr, X_indices, y):
        return spectral_norm(X_data, X_indptr, X_indices, len(y)) ** 2 / len(y)

    def value(self, y, w, Xw):
        return np.sum((y - Xw) ** 2) / (2 * len(Xw))

    def gradient_scalar(self, X, y, w, Xw, j):
        return (X[:, j] @ Xw - self.Xty[j]) / len(Xw)

    def gradient_scalar_sparse(self, X_data, X_indptr, X_indices, y, Xw, j):
        XjTXw = 0.
        for i in range(X_indptr[j], X_indptr[j+1]):
            XjTXw += X_data[i] * Xw[X_indices[i]]
        return (XjTXw - self.Xty[j]) / len(Xw)

    def gradient(self, X, y, Xw):
        return X.T @ (Xw - y) / len(y)

    def raw_grad(self, y, Xw):
        """Compute gradient of datafit w.r.t ``Xw``."""
        return (Xw - y) / len(y)

    def raw_hessian(self, y, Xw):
        """Compute Hessian of datafit w.r.t ``Xw``."""
        return np.ones(len(y)) / len(y)

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


class WeightedQuadratic(BaseDatafit):
    r"""Weighted Quadratic datafit to handle sample weights.

    The datafit reads:

    .. math:: 1 / (2 xx  \sum_(i=1)^(n_"samples") weights_i)
        \sum_(i=1)^(n_"samples") weights_i (y_i - (Xw)_i)^ 2

    Attributes
    ----------
    Xtwy : array, shape (n_features,)
        Pre-computed quantity used during the gradient evaluation.
        Equal to ``X.T @ (samples_weights * y)``.
    sample_weights : array, shape (n_samples,)
        Weights for each sample.

    Note
    ----
    The class is jit compiled at fit time using Numba compiler.
    This allows for faster computations.
    """

    def __init__(self, sample_weights):
        self.sample_weights = sample_weights

    def get_spec(self):
        spec = (
            ('Xtwy', float64[:]),
            ('sample_weights', float64[:]),
        )
        return spec

    def params_to_dict(self):
        return {'sample_weights': self.sample_weights}

    def get_lipschitz(self, X, y):
        n_features = X.shape[1]
        lipschitz = np.zeros(n_features, dtype=X.dtype)
        w_sum = self.sample_weights.sum()

        for j in range(n_features):
            lipschitz[j] = (self.sample_weights * X[:, j] ** 2).sum() / w_sum

        return lipschitz

    def get_lipschitz_sparse(self, X_data, X_indptr, X_indices, y):
        n_features = len(X_indptr) - 1
        lipschitz = np.zeros(n_features, dtype=X_data.dtype)
        w_sum = self.sample_weights.sum()

        for j in range(n_features):
            nrm2 = 0.
            for idx in range(X_indptr[j], X_indptr[j + 1]):
                nrm2 += self.sample_weights[X_indices[idx]] * X_data[idx] ** 2

            lipschitz[j] = nrm2 / w_sum

        return lipschitz

    def initialize(self, X, y):
        self.Xtwy = X.T @ (self.sample_weights * y)

    def initialize_sparse(self, X_data, X_indptr, X_indices, y):
        n_features = len(X_indptr) - 1
        self.Xty = np.zeros(n_features, dtype=X_data.dtype)

        for j in range(n_features):
            xty = 0
            for idx in range(X_indptr[j], X_indptr[j + 1]):
                xty += (X_data[idx] * self.sample_weights[X_indices[idx]]
                        * y[X_indices[idx]])
            self.Xty[j] = xty

    def get_global_lipschitz(self, X, y):
        w_sum = self.sample_weights.sum()
        return norm(X.T @ np.sqrt(self.sample_weights), ord=2) ** 2 / w_sum

    def get_global_lipschitz_sparse(self, X_data, X_indptr, X_indices, y):
        return spectral_norm(
            X_data * np.sqrt(self.sample_weights[X_indices]),
            X_indptr, X_indices, len(y)) ** 2 / self.sample_weights.sum()

    def value(self, y, w, Xw):
        w_sum = self.sample_weights.sum()
        return np.sum(self.sample_weights * (y - Xw) ** 2) / (2 * w_sum)

    def gradient_scalar(self, X, y, w, Xw, j):
        return (X[:, j] @ (self.sample_weights * (Xw - y))) / self.sample_weights.sum()

    def gradient_scalar_sparse(self, X_data, X_indptr, X_indices, y, Xw, j):
        XjTXw = 0.
        for i in range(X_indptr[j], X_indptr[j + 1]):
            XjTXw += X_data[i] * self.sample_weights[X_indices[i]] * Xw[X_indices[i]]
        return (XjTXw - self.Xty[j]) / self.sample_weights.sum()

    def gradient(self, X, y, Xw):
        return X.T @ (self.sample_weights * (Xw - y)) / self.sample_weights.sum()

    def raw_grad(self, y, Xw):
        return (self.sample_weights * (Xw - y)) / self.sample_weights.sum()

    def raw_hessian(self, y, Xw):
        return self.sample_weights / self.sample_weights.sum()

    def full_grad_sparse(self, X_data, X_indptr, X_indices, y, Xw):
        n_features = X_indptr.shape[0] - 1
        grad = np.zeros(n_features, dtype=Xw.dtype)

        for j in range(n_features):
            XjTXw = 0.
            for i in range(X_indptr[j], X_indptr[j + 1]):
                XjTXw += (X_data[i] * self.sample_weights[X_indices[i]]
                          * Xw[X_indices[i]])
            grad[j] = (XjTXw - self.Xty[j]) / self.sample_weights.sum()
        return grad

    def intercept_update_step(self, y, Xw):
        return np.sum(self.sample_weights * (Xw - y)) / self.sample_weights.sum()


class QuadraticHessian(BaseDatafit):
    r"""Quadratic datafit where we pass the Hessian A directly.

    The datafit reads:

    .. math:: 1 / 2 x^(\top) A x + \langle b, x \rangle

    For a symmetric A. Up to a constant, it is the same as a Quadratic, with
    :math:`A = 1 / (n_"samples") X^(\top)X` and :math:`b = - 1 / n_"samples" X^(\top)y`.
    When the Hessian is available, this datafit is more efficient than using Quadratic.
    """

    def __init__(self):
        pass

    def get_spec(self):
        pass

    def params_to_dict(self):
        return dict()

    def get_lipschitz(self, A, b):
        n_features = A.shape[0]
        lipschitz = np.zeros(n_features, dtype=A.dtype)
        for j in range(n_features):
            lipschitz[j] = A[j, j]
        return lipschitz

    def gradient_scalar(self, A, b, w, Ax, j):
        return Ax[j] + b[j]

    def value(self, b, x, Ax):
        return 0.5 * (x*Ax).sum() + (b*x).sum()


@njit
def sigmoid(x):
    """Vectorwise sigmoid."""
    out = 1 / (1 + np.exp(- x))
    return out


class Logistic(BaseDatafit):
    r"""Logistic datafit with labels in {-1, 1}.

    The datafit reads:

    .. math:: 1 / n_"samples" \sum_(i=1)^(n_"samples") log(1 + exp(-y_i (Xw)_i))

    Notes
    -----
    The class is jit compiled at fit time using Numba compiler.
    This allows for faster computations.
    """

    def __init__(self):
        pass

    def get_spec(self):
        pass

    def params_to_dict(self):
        return dict()

    def raw_grad(self, y, Xw):
        """Compute gradient of datafit w.r.t ``Xw``."""
        return -y / (1 + np.exp(y * Xw)) / len(y)

    def raw_hessian(self, y, Xw):
        """Compute Hessian of datafit w.r.t ``Xw``."""
        exp_minus_yXw = np.exp(-y * Xw)
        return exp_minus_yXw / (1 + exp_minus_yXw) ** 2 / len(y)

    def get_lipschitz(self, X, y):
        return (X ** 2).sum(axis=0) / (4 * len(y))

    def get_lipschitz_sparse(self, X_data, X_indptr, X_indices, y):
        n_features = len(X_indptr) - 1

        lipschitz = np.zeros(n_features, dtype=X_data.dtype)
        for j in range(n_features):
            Xj = X_data[X_indptr[j]:X_indptr[j+1]]
            lipschitz[j] = (Xj ** 2).sum() / (len(y) * 4)

        return lipschitz

    def get_global_lipschitz(self, X, y):
        return norm(X, ord=2) ** 2 / (4 * len(y))

    def get_global_lipschitz_sparse(self, X_data, X_indptr, X_indices, y):
        return spectral_norm(
            X_data, X_indptr, X_indices, len(y)) ** 2 / (4 * len(y))

    def value(self, y, w, Xw):
        return np.log(1. + np.exp(- y * Xw)).sum() / len(y)

    def gradient_scalar(self, X, y, w, Xw, j):
        return (- X[:, j] @ (y * sigmoid(- y * Xw))) / len(y)

    def gradient(self, X, y, Xw):
        return X.T @ self.raw_grad(y, Xw)

    def gradient_sparse(self, X_data, X_indptr, X_indices, y, Xw):
        n_features = X_indptr.shape[0] - 1
        out = np.zeros(n_features, dtype=X_data.dtype)
        raw_grad = self.raw_grad(y, Xw)

        for j in range(n_features):
            out[j] = _sparse_xj_dot(X_data, X_indptr, X_indices, j, raw_grad)

        return out

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

    The datafit reads:

    .. math:: 1/2 ||(yX)^T w||_2 ^ 2

    Notes
    -----
    The class is jit compiled at fit time using Numba compiler.
    This allows for faster computations.
    """

    def __init__(self):
        pass

    def get_spec(self):
        pass

    def params_to_dict(self):
        return dict()

    def get_lipschitz(self, yXT, y):
        n_features = yXT.shape[1]
        lipschitz = np.zeros(n_features, dtype=yXT.dtype)

        for j in range(n_features):
            lipschitz[j] = norm(yXT[:, j]) ** 2

        return lipschitz

    def get_lipschitz_sparse(self, yXT_data, yXT_indptr, yXT_indices, y):
        n_features = len(yXT_indptr) - 1

        lipschitz = np.zeros(n_features, dtype=yXT_data.dtype)
        for j in range(n_features):
            nrm2 = 0.
            for idx in range(yXT_indptr[j], yXT_indptr[j + 1]):
                nrm2 += yXT_data[idx] ** 2
            lipschitz[j] = nrm2

        return lipschitz

    def get_global_lipschitz(self, yXT, y):
        return norm(yXT, ord=2) ** 2

    def get_global_lipschitz_sparse(self, yXT_data, yXT_indptr, yXT_indices, y):
        return spectral_norm(
            yXT_data, yXT_indptr, yXT_indices, max(yXT_indices)+1) ** 2

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

    The datafit reads:

    .. math:: 1 / n_"samples" sum_(i=1)^(n_"samples") f(y_i - (Xw)_i)

    where :math:`f` is the Huber function:

    .. math::
        f(x) = {(1/2 x^2                   , if x <= delta),
                (delta abs(x) - 1/2 delta^2, if x > delta):}

    Attributes
    ----------
    delta : float
        Threshold hyperparameter.

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
        )
        return spec

    def params_to_dict(self):
        return dict(delta=self.delta)

    def get_lipschitz(self, X, y):
        n_features = X.shape[1]

        lipschitz = np.zeros(n_features, dtype=X.dtype)
        for j in range(n_features):
            lipschitz[j] = (X[:, j] ** 2).sum() / len(y)

        return lipschitz

    def get_lipschitz_sparse(self, X_data, X_indptr, X_indices, y):
        n_features = len(X_indptr) - 1

        lipschitz = np.zeros(n_features, dtype=X_data.dtype)
        for j in range(n_features):
            nrm2 = 0.
            for idx in range(X_indptr[j], X_indptr[j + 1]):
                nrm2 += X_data[idx] ** 2
            lipschitz[j] = nrm2 / len(y)

        return lipschitz

    def get_global_lipschitz(self, X, y):
        return norm(X, ord=2) ** 2 / len(y)

    def get_global_lipschitz_sparse(self, X_data, X_indptr, X_indices, y):
        return spectral_norm(
            X_data, X_indptr, X_indices, len(y)) ** 2 / len(y)

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

    The datafit reads:

    .. math:: 1 / n_"samples" sum_(i=1)^(n_"samples") (exp((Xw)_i) - y_i (Xw)_i)

    Notes
    -----
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
        if np.any(y < 0):
            raise ValueError(
                "Target vector `y` should only take positive values "
                "when fitting a Poisson model.")

    def initialize_sparse(self, X_data, X_indptr, X_indices, y):
        if np.any(y < 0):
            raise ValueError(
                "Target vector `y` should only take positive values "
                "when fitting a Poisson model.")

    def raw_grad(self, y, Xw):
        """Compute gradient of datafit w.r.t ``Xw``."""
        return (np.exp(Xw) - y) / len(y)

    def raw_hessian(self, y, Xw):
        """Compute Hessian of datafit w.r.t ``Xw``."""
        return np.exp(Xw) / len(y)

    def value(self, y, w, Xw):
        return np.sum(np.exp(Xw) - y * Xw) / len(y)

    def gradient(self, X, y, Xw):
        return X.T @ self.raw_grad(y, Xw)

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

    def intercept_update_step(self, y, Xw):
        return np.sum(self.raw_grad(y, Xw))


class Gamma(BaseDatafit):
    r"""Gamma datafit.

    The datafit reads:

    .. math::
        1 / n_"samples" \sum_(i=1)^(n_"samples")
        ((Xw)_i + y_i exp(-(Xw)_i) - 1 - log(y_i))

    Notes
    -----
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
        if np.any(y <= 0):
            raise ValueError(
                "Target vector `y` should only take positive values "
                "when fitting a Gamma model.")

    def initialize_sparse(self, X_data, X_indptr, X_indices, y):
        if np.any(y <= 0):
            raise ValueError(
                "Target vector `y` should only take positive values "
                "when fitting a Gamma model.")

    def raw_grad(self, y, Xw):
        """Compute gradient of datafit w.r.t. ``Xw``."""
        return (1 - y * np.exp(-Xw)) / len(y)

    def raw_hessian(self, y, Xw):
        """Compute Hessian of datafit w.r.t. ``Xw``."""
        return (y * np.exp(-Xw)) / len(y)

    def value(self, y, w, Xw):
        return (np.sum(Xw + y * np.exp(-Xw) - np.log(y)) - 1) / len(y)

    def gradient_scalar(self, X, y, w, Xw, j):
        return X[:, j] @ (1 - y * np.exp(-Xw)) / len(y)

    def intercept_update_step(self, y, Xw):
        return np.sum(self.raw_grad(y, Xw))


class Cox(BaseDatafit):
    r"""Cox datafit for survival analysis.

    Refer to :ref:`Mathematics behind Cox datafit <maths_cox_datafit>` for details.

    Parameters
    ----------
    use_efron : bool, default=False
        If ``True`` uses Efron estimate to handle tied observations.

    Attributes
    ----------
    T_indices : array-like, shape (n_samples,)
        Indices of observations with the same occurrence times stacked horizontally as
        ``[group_1, group_2, ...]`` in ascending order. It is initialized
        with the ``.initialize`` method (or ``initialize_sparse`` for sparse ``X``).

    T_indptr : array-like, (np.unique(tm) + 1,)
        Array where two consecutive elements delimit a group of
        observations having the same occurrence times.

    H_indices : array-like, shape (n_samples,)
        Indices of uncensored observations with the same occurrence times stacked
        horizontally as ``[group_1, group_2, ...]`` in ascending order.
        It is initialized when calling the ``.initialize`` method
        (or ``initialize_sparse`` for sparse ``X``) when ``use_efron=True``.

    H_indptr : array-like, shape (np.unique(tm[s != 0]) + 1,)
        Array where two consecutive elements delimits a group of uncensored
        observations having the same occurrence time.
    """

    def __init__(self, use_efron=False):
        self.use_efron = use_efron

    def get_spec(self):
        return (
            ('use_efron', bool_),
            ('T_indptr', int64[:]), ('T_indices', int64[:]),
            ('H_indptr', int64[:]), ('H_indices', int64[:]),
        )

    def params_to_dict(self):
        return dict(use_efron=self.use_efron)

    def value(self, y, w, Xw):
        """Compute the value of the datafit."""
        tm, s = y[:, 0], y[:, 1]  # noqa
        n_samples = Xw.shape[0]

        # compute inside log term
        exp_Xw = np.exp(Xw)
        B_exp_Xw = self._B_dot_vec(exp_Xw)
        if self.use_efron:
            B_exp_Xw -= self._A_dot_vec(exp_Xw)

        out = -(s @ Xw) + s @ np.log(B_exp_Xw)
        return out / n_samples

    def raw_grad(self, y, Xw):
        r"""Compute gradient of datafit w.r.t. ``Xw``.

        Refer to :ref:`Mathematics behind Cox datafit <maths_cox_datafit>`
        equation 4 for details.
        """
        tm, s = y[:, 0], y[:, 1]  # noqa
        n_samples = Xw.shape[0]

        exp_Xw = np.exp(Xw)
        B_exp_Xw = self._B_dot_vec(exp_Xw)
        if self.use_efron:
            B_exp_Xw -= self._A_dot_vec(exp_Xw)

        s_over_B_exp_Xw = s / B_exp_Xw
        out = -s + exp_Xw * self._B_T_dot_vec(s_over_B_exp_Xw)
        if self.use_efron:
            out -= exp_Xw * self._AT_dot_vec(s_over_B_exp_Xw)

        return out / n_samples

    def raw_hessian(self, y, Xw):
        """Compute a diagonal upper bound of the datafit's Hessian w.r.t. ``Xw``.

        Refer to :ref:`Mathematics behind Cox datafit <maths_cox_datafit>`
        equation 6 for details.
        """
        tm, s = y[:, 0], y[:, 1]  # noqa
        n_samples = Xw.shape[0]

        exp_Xw = np.exp(Xw)
        B_exp_Xw = self._B_dot_vec(exp_Xw)
        if self.use_efron:
            B_exp_Xw -= self._A_dot_vec(exp_Xw)

        s_over_B_exp_Xw = s / B_exp_Xw
        out = exp_Xw * self._B_T_dot_vec(s_over_B_exp_Xw)
        if self.use_efron:
            out -= exp_Xw * self._AT_dot_vec(s_over_B_exp_Xw)

        return out / n_samples

    def gradient(self, X, y, Xw):
        """Compute gradient of the datafit."""
        return X.T @ self.raw_grad(y, Xw)

    def gradient_sparse(self, X_data, X_indptr, X_indices, y, Xw):
        """Compute gradient of the datafit in case ``X`` is sparse."""
        n_features = X_indptr.shape[0] - 1
        out = np.zeros(n_features, dtype=X_data.dtype)
        raw_grad = self.raw_grad(y, Xw)

        for j in range(n_features):
            out[j] = _sparse_xj_dot(X_data, X_indptr, X_indices, j, raw_grad)

        return out

    def initialize(self, X, y):
        """Initialize the datafit attributes."""
        tm, s = y[:, 0], y[:, 1]  # noqa

        self.T_indices = np.argsort(tm)
        self.T_indptr = self._get_indptr(tm, self.T_indices)

        if self.use_efron:
            # filter out censored data
            self.H_indices = self.T_indices[s[self.T_indices] != 0]
            self.H_indptr = self._get_indptr(tm, self.H_indices)

    def initialize_sparse(self, X_data, X_indptr, X_indices, y):
        """Initialize the datafit attributes in sparse dataset case."""
        # `initialize_sparse` and `initialize` have the same implementation
        # small hack to avoid repetitive code: pass in X_data as only its dtype is used
        self.initialize(X_data, y)

    def get_global_lipschitz(self, X, y):
        s = y[:, 1]

        n_samples = X.shape[0]
        return s.sum() * norm(X, ord=2) ** 2 / n_samples

    def get_global_lipschitz_sparse(self, X_data, X_indptr, X_indices, y):
        s = y[:, 1]

        n_samples = s.shape[0]
        norm_X = spectral_norm(X_data, X_indptr, X_indices, n_samples)

        return s.sum() * norm_X ** 2 / n_samples

    def _B_dot_vec(self, vec):
        # compute `B @ vec` in O(n) instead of O(n^2)
        out = np.zeros_like(vec)
        n_T = self.T_indptr.shape[0] - 1
        cum_sum = 0.

        # reverse loop to avoid starting from cum_sum and subtracting vec coordinates
        # subtracting big numbers results in 'cancellation errors' and hence erroneous
        # results. Ref. J Nocedal, "Numerical optimization", page 615
        for idx in range(n_T - 1, -1, -1):
            current_T_idx = self.T_indices[self.T_indptr[idx]: self.T_indptr[idx+1]]

            cum_sum += np.sum(vec[current_T_idx])
            out[current_T_idx] = cum_sum

        return out

    def _B_T_dot_vec(self, vec):
        # compute `B.T @ vec` in O(n) instead of O(n^2)
        out = np.zeros_like(vec)
        n_T = self.T_indptr.shape[0] - 1
        cum_sum = 0.

        for idx in range(n_T):
            current_T_idx = self.T_indices[self.T_indptr[idx]: self.T_indptr[idx+1]]

            cum_sum += np.sum(vec[current_T_idx])
            out[current_T_idx] = cum_sum

        return out

    def _A_dot_vec(self, vec):
        # compute `A @ vec` in O(n) instead of O(n^2)
        out = np.zeros_like(vec)
        n_H = self.H_indptr.shape[0] - 1

        for idx in range(n_H):
            current_H_idx = self.H_indices[self.H_indptr[idx]: self.H_indptr[idx+1]]
            size_current_H = current_H_idx.shape[0]
            frac_range = np.arange(size_current_H, dtype=vec.dtype) / size_current_H

            sum_vec_H = np.sum(vec[current_H_idx])
            out[current_H_idx] = sum_vec_H * frac_range

        return out

    def _AT_dot_vec(self, vec):
        # compute `A.T @ vec` in O(n) instead of O(n^2)
        out = np.zeros_like(vec)
        n_H = self.H_indptr.shape[0] - 1

        for idx in range(n_H):
            current_H_idx = self.H_indices[self.H_indptr[idx]: self.H_indptr[idx+1]]
            size_current_H = current_H_idx.shape[0]
            frac_range = np.arange(size_current_H, dtype=vec.dtype) / size_current_H

            weighted_sum_vec_H = vec[current_H_idx] @ frac_range
            out[current_H_idx] = weighted_sum_vec_H * np.ones(size_current_H)

        return out

    def _get_indptr(self, vals, indices):
        # given `indices = argsort(vals)`
        # build and array `indptr` where two consecutive elements
        # delimit indices with the same val
        n_indices = indices.shape[0]

        indptr = [0]
        count = 1
        for i in range(n_indices - 1):
            if vals[indices[i]] == vals[indices[i+1]]:
                count += 1
            else:
                indptr.append(count + indptr[-1])
                count = 1
        indptr.append(n_indices)

        return np.asarray(indptr, dtype=np.int64)
