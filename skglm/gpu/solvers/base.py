from numba import njit
from abc import abstractmethod

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spicy_linalg

from skglm.utils.prox_funcs import ST_vec


class BaseFistaSolver:

    @abstractmethod
    def solve(self, X, y, datafit, penalty):
        ...


class BaseQuadratic:

    def value(self, X, y, w):
        """parameters are numpy/scipy arrays."""
        return ((y - X @ w) ** 2).sum() / len(y)

    def gradient(self, X, y, w, Xw):
        return X.T @ (Xw - y) / len(y)

    def get_lipschitz_cst(self, X):
        n_samples = len(X)

        if sparse.issparse(X):
            return spicy_linalg.svds(X, k=1)[1][0] ** 2 / n_samples

        return np.linalg.norm(X, ord=2) ** 2 / n_samples


class BaseL1:

    def __init__(self, alpha):
        self.alpha = alpha

    def prox(self, value, stepsize):
        return ST_vec(value, self.alpha * stepsize)

    def max_subdiff_distance(self, w, grad):
        return BaseL1._compute_dist_subdiff(w, grad, self.alpha)

    @staticmethod
    @njit("f8(f8[:], f8[:], f8)")
    def _compute_max_subdiff_distance(w, grad, lmbd):
        max_dist = 0.

        for i in range(len(w)):
            grad_i = grad[i]
            w_i = w[i]

            if w[i] == 0.:
                dist = max(abs(grad_i) - lmbd, 0.)
            else:
                dist = abs(grad_i + np.sign(w_i) * lmbd)

            max_dist = max(max_dist, dist)

        return max_dist
