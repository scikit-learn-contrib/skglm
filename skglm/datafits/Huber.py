import numpy as np
from numba import float64
from numba.experimental import jitclass
from numba.types import bool_
from skglm.datafits import BaseDatafit
from skglm.penalties import BasePenalty

spec_L1R = [
    ('alpha', float64)
]


@jitclass(spec_L1R)
class L1_R(BasePenalty):
    """L1 penalty without the first term."""

    def __init__(self, alpha):
        self.alpha = alpha

    def value(self, w):
        """Compute L1 penalty value without the first term."""
        return self.alpha * np.sum(np.abs(w[1:]))

    def prox_1d(self, value, stepsize, j):
        """Compute proximal operator of the L1 penalty (soft-thresholding operator)."""
        prox = value
        w0 = prox[0]
        prox -= np.sign(value) * abs(np.clip(value, -self.alpha * stepsize,
                                             self.alpha * stepsize))
        prox[0] = w0
        return prox

    def subdiff_distance(self, w, grad, ws):
        """Compute distance of negative gradient to the subdifferential at w."""
        subdiff_dist = np.zeros_like(grad)
        for idx, j in enumerate(ws):
            if w[j] == 0:
                # distance of - grad_j to  [-alpha, alpha]
                subdiff_dist[idx] = max(0, np.abs(grad[idx]) - self.alpha)
            else:
                # distance of - grad_j to alpha * sign(w[j])
                subdiff_dist[idx] = np.abs(
                    - grad[idx] - np.sign(w[j]) * self.alpha)
        subdiff_dist[0] = 0
        return subdiff_dist

    def is_penalized(self, n_features):
        """Return a binary mask with the penalized features."""
        return np.ones(n_features, bool_)

    def generalized_support(self, w):
        """Return a mask with non-zero coefficients."""
        return w != 0

    def alpha_max(self, gradient0):
        """Return penalization value for which 0 is solution."""
        return np.max(np.abs(gradient0))


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


if __name__ == '__main__':
    from skglm import GeneralizedLinearEstimator

    clf = GeneralizedLinearEstimator(
        Huber(0.1), L1_R(0.1), is_classif=False
    )
    X = np.random.randn(10, 3)
    y = np.random.randn(10)
    clf.fit(X, y)
