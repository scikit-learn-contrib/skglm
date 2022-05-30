import numpy as np
from numpy.linalg import norm
from numba.experimental import jitclass
from numba import int32, float64

from skglm.datafits.base import BaseDatafit


spec_QuadraticGroup = [
    ('grp_partition', int32[:]),
    ('grp_indices', int32[:]),
    ('lipschitz', float64[:])
]


@ jitclass(spec_QuadraticGroup)
class QuadraticGroup(BaseDatafit):
    """Quadratic datafit used with group penalties.

    The datafit reads::

    (1 / (2 * n_samples)) * ||y - X w||^2_2

    where coordinates of ``w`` admits a group partitions.

    Attributes
    ----------
    grp_partition : array, shape (n_groups + 1,)
        The group partition.

    grp_indices : array, shape (n_features,)
        The group indices.

    lipschitz : array, shape (n_groups,)
        The lipschitz constants for each group.
    """

    def __init__(self, grp_partition, grp_indices):
        self.grp_partition, self.grp_indices = grp_partition, grp_indices

    def initialize(self, X, y):
        n_samples = len(y)
        grp_partition, grp_indices = self.grp_partition, self.grp_indices
        n_groups = len(grp_partition) - 1

        lipschitz = np.zeros(n_groups)
        for g in range(n_groups):
            grp_g_indices = grp_indices[grp_partition[g]: grp_partition[g+1]]
            X_g = X[:, grp_g_indices]

            lipschitz[g] = norm(X_g) ** 2 / n_samples

        self.lipschitz = lipschitz

    def value(self, y, w, Xw):
        n_samples = len(y)
        return 1 / (2*n_samples) * norm(y - Xw) ** 2

    def gradient_g(self, X, y, w, Xw, g):
        grp_partition, grp_indices = self.grp_partition, self.grp_indices
        grp_g_indices = grp_indices[grp_partition[g]: grp_partition[g+1]]

        grad_g = np.zeros(len(grp_g_indices))
        for i, g in enumerate(grp_g_indices):
            grad_g[i] = self.gradient_scalar(X, y, w, Xw, g)

        return grad_g

    def gradient_scalar(self, X, y, w, Xw, j):
        n_samples = len(y)
        return - 1 / n_samples * X[:, j].T @ (y - Xw)
