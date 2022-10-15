import numpy as np
from numpy.linalg import norm
from numba import int32, float64

from skglm.datafits.base import BaseDatafit


class QuadraticGroup(BaseDatafit):
    r"""Quadratic datafit used with group penalties.

    The datafit reads::

    (1 / (2 * n_samples)) * ||y - X w||^2_2

    Attributes
    ----------
    grp_indices : array, shape (n_features,)
        The group indices stacked contiguously
        ([grp1_indices, grp2_indices, ...]).

    grp_ptr : array, shape (n_groups + 1,)
        The group pointers such that two consecutive elements delimit
        the indices of a group in ``grp_indices``.

    lipschitz : array, shape (n_groups,)
        The lipschitz constants for each group.
    """

    def __init__(self, grp_ptr, grp_indices):
        self.grp_ptr, self.grp_indices = grp_ptr, grp_indices

    def get_spec(self):
        spec = (
            ('grp_ptr', int32[:]),
            ('grp_indices', int32[:]),
            ('lipschitz', float64[:])
        )
        return spec

    def params_to_dict(self):
        return dict(grp_ptr=self.grp_ptr,
                    grp_indices=self.grp_indices)

    def initialize(self, X, y):
        grp_ptr, grp_indices = self.grp_ptr, self.grp_indices
        n_groups = len(grp_ptr) - 1

        lipschitz = np.zeros(n_groups)
        for g in range(n_groups):
            grp_g_indices = grp_indices[grp_ptr[g]: grp_ptr[g+1]]
            X_g = X[:, grp_g_indices]
            lipschitz[g] = norm(X_g, ord=2) ** 2 / len(y)

        self.lipschitz = lipschitz

    def value(self, y, w, Xw):
        return norm(y - Xw) ** 2 / (2 * len(y))

    def gradient_g(self, X, y, w, Xw, g):
        grp_ptr, grp_indices = self.grp_ptr, self.grp_indices
        grp_g_indices = grp_indices[grp_ptr[g]: grp_ptr[g+1]]

        grad_g = np.zeros(len(grp_g_indices))
        for idx, j in enumerate(grp_g_indices):
            grad_g[idx] = self.gradient_scalar(X, y, w, Xw, j)

        return grad_g

    def gradient_scalar(self, X, y, w, Xw, j):
        return X[:, j] @ (Xw - y) / len(y)

    def intercept_update_step(self, y, Xw):
        return np.mean(Xw - y)


class LogisticGroup(BaseDatafit):
    r"""Logistic datafit used with group penalties.

    The datafit reads::

    (1 / n_samples) * \sum_i log(1 + exp(-y_i * Xw_i))

    Attributes
    ----------
    grp_indices : array, shape (n_features,)
        The group indices stacked contiguously
        ([grp1_indices, grp2_indices, ...]).

    grp_ptr : array, shape (n_groups + 1,)
        The group pointers such that two consecutive elements delimit
        the indices of a group in ``grp_indices``.

    lipschitz : array, shape (n_groups,)
        The lipschitz constants for each group.
    """

    def __init__(self, grp_ptr, grp_indices):
        self.grp_ptr, self.grp_indices = grp_ptr, grp_indices

    def get_spec(self):
        spec = (
            ('grp_ptr', int32[:]),
            ('grp_indices', int32[:]),
            ('lipschitz', float64[:])
        )
        return spec

    def params_to_dict(self):
        return dict(grp_ptr=self.grp_ptr,
                    grp_indices=self.grp_indices)

    def initialize(self, X, y):
        grp_ptr, grp_indices = self.grp_ptr, self.grp_indices
        n_groups = len(grp_ptr) - 1

        lipschitz = np.zeros(n_groups)
        for g in range(n_groups):
            grp_g_indices = grp_indices[grp_ptr[g]: grp_ptr[g+1]]
            X_g = X[:, grp_g_indices]
            lipschitz[g] = norm(X_g, ord=2) ** 2 / (4 * len(y))

        self.lipschitz = lipschitz

    def value(self, y, w, Xw):
        return np.log(1 + np.exp(-y * Xw)).sum() / len(y)

    def raw_grad(self, y, Xw):
        """Compute gradient of datafit w.r.t ``Xw``."""
        return -y / (1 + np.exp(y * Xw)) / len(y)

    def raw_hessian(self, y, Xw):
        """Compute Hessian of datafit w.r.t ``Xw``."""
        exp_minus_yXw = np.exp(-y * Xw)
        return exp_minus_yXw / (1 + exp_minus_yXw) ** 2 / len(y)

    def gradient_g(self, X, y, w, Xw, g):
        grp_ptr, grp_indices = self.grp_ptr, self.grp_indices
        grp_g_indices = grp_indices[grp_ptr[g]: grp_ptr[g+1]]
        raw_grad_val = self.raw_grad(y, Xw)

        grad_g = np.zeros(len(grp_g_indices))
        for idx, j in enumerate(grp_g_indices):
            grad_g[idx] = X[:, j] @ raw_grad_val

        return grad_g

    def intercept_update_step(self, y, Xw):
        return np.mean(self.raw_grad(y, Xw)) / 4
