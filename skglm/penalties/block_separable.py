import numpy as np
from numpy.linalg import norm
from numba import float64, int64
from numba.experimental import jitclass
from numba.types import bool_
from pkg_resources import cleanup_resources

from skglm.penalties.base import BasePenalty
from skglm.utils import BST, BST_vec, prox_block_2_05, ST_vec


spec_L21 = [
    ('alpha', float64)
]


@jitclass(spec_L21)
class L2_1(BasePenalty):
    """L2/1 row-wise penalty: sum of L2 norms of rows."""

    def __init__(self, alpha):
        self.alpha = alpha

    def value(self, W):
        """Compute the L2/1 penalty value."""
        return self.alpha * np.sqrt(np.sum(W ** 2, axis=1)).sum()

    def prox_1feat(self, value, stepsize, j):
        """Compute proximal operator of the L2/1 penalty (block soft thresholding)."""
        return BST(value, self.alpha * stepsize)

    def subdiff_distance(self, W, grad, ws):
        """Compute distance of negative gradient to the subdifferential at W."""
        subdiff_dist = np.zeros_like(ws, dtype=grad.dtype)
        for idx, j in enumerate(ws):
            if not np.any(W[j, :]):
                # distance of - grad_j to alpha * the unit l2 ball
                norm_grad_j = norm(grad[idx, :])
                subdiff_dist[idx] = max(0, norm_grad_j - self.alpha)
            else:
                # distance of -grad_j to alpha * W[j] / norm(W[j])
                subdiff_dist[idx] = norm(
                    grad[idx, :]
                    + self.alpha * W[j, :] / norm(W[j, :]))
        return subdiff_dist

    def is_penalized(self, n_features):
        """Return a binary mask with the penalized features."""
        return np.ones(n_features, bool_)


spec_L2_05 = [
    ('alpha', float64)
]


@jitclass(spec_L2_05)
class L2_05(BasePenalty):
    """L2/0.5 row-wise penalty: sum of square roots of L2 norms of rows."""

    def __init__(self, alpha):
        self.alpha = alpha

    def value(self, W):
        """Compute the value of L2/0.5 at w."""
        W_rows_norm = np.zeros(W.shape[0])
        for i in range(W.shape[0]):
            W_rows_norm[i] = norm(W[i])
        return self.alpha * np.sum(np.sqrt(W_rows_norm))

    def prox_1feat(self, value, stepsize, j):
        """Compute the proximal operator of L2/0.5."""
        return prox_block_2_05(value, self.alpha * stepsize)

    def subdiff_distance(self, W, grad, ws):
        """Compute distance of negative gradient to the subdifferential at W."""
        subdiff_dist = np.zeros_like(ws, dtype=grad.dtype)
        for idx, j in enumerate(ws):
            if not np.any(W[j, :]):
                subdiff_dist[idx] = 0.
            else:
                subdiff_dist[idx] = norm(
                    grad[idx, :] + self.alpha * W[j, :] / (2 * norm(W[j, :])**(3./2.))
                )
        return subdiff_dist

    def is_penalized(self, n_features):
        """Return a binary mask with the penalized features."""
        return np.ones(n_features, bool_)


spec_BlockMCPenalty = [
    ('alpha', float64),
    ('gamma', float64),
]


@jitclass(spec_BlockMCPenalty)
class BlockMCPenalty(BasePenalty):
    """Block Minimax Concave Penalty.

    Notes
    -----
    With W_j the j-th row of W, the penalty is:
        pen(||W_j||) = alpha * ||W_j|| - ||W_j||^2 / (2 * gamma)
                       if ||W_j|| =< gamma * alpha
                     = gamma * alpha ** 2 / 2
                       if ||W_j|| > gamma * alpha

        value = sum_{j=1}^{n_features} pen(||W_j||)
    """

    def __init__(self, alpha, gamma):
        self.alpha = alpha
        self.gamma = gamma

    def value(self, W):
        """Compute the value of BlockMCP at W."""
        norm_rows = np.sqrt(np.sum(W ** 2, axis=1))
        s0 = norm_rows < self.gamma * self.alpha
        value = np.full_like(norm_rows, self.gamma * self.alpha ** 2 / 2.)
        value[s0] = self.alpha * norm_rows[s0] - norm_rows[s0]**2 / (2 * self.gamma)
        return np.sum(value)

    def prox_1feat(self, value, stepsize, j):
        """Compute the proximal operator of BlockMCP."""
        tau = self.alpha * stepsize
        g = self.gamma / stepsize
        norm_value = norm(value)
        if norm_value <= tau:
            return np.zeros_like(value)
        if norm_value > g * tau:
            return value
        return (1 - tau / norm_value) * value / (1. - 1./g)

    def subdiff_distance(self, W, grad, ws):
        """Compute distance of negative gradient to the subdifferential at W."""
        subdiff_dist = np.zeros_like(ws, dtype=grad.dtype)
        for idx, j in enumerate(ws):
            norm_Wj = norm(W[j])
            if not np.any(W[j]):
                # distance of -grad_j to alpha * unit ball
                norm_grad_j = norm(grad[idx])
                subdiff_dist[idx] = max(0, norm_grad_j - self.alpha)
            elif norm_Wj < self.alpha * self.gamma:
                # distance of -grad_j to alpha * W[j] / ||W_j|| -  W[j] / gamma
                subdiff_dist[idx] = norm(
                    grad[idx] + self.alpha * W[j]/norm_Wj - W[j] / self.gamma)
            else:
                # distance of -grad to 0
                subdiff_dist[idx] = norm(grad[idx])
        return subdiff_dist

    def is_penalized(self, n_features):
        """Return a binary mask with the penalized features."""
        return np.ones(n_features, bool_)


spec_SparseGroupL1 = {
    'alpha': float64,
    'tau': float64,
    'weights': float64[:]
}


@jitclass(spec_SparseGroupL1)
class SparseGroupL1(BasePenalty):
    r"""Sparse Group L1 penalty.

    :math::

        \alpha (\tau \Vert \beta \Vert_1 
        + (1-\tau) \sum_g \omega_g \Vert \beta_g \Vert_2)

    where :math:`\beta` is the vector of coefficients, :math:`\omega_g` are
    the weights of the groups, :math:`\alpha` the intensity of the penalty, 
    and :math:`\tau` the ratio individual/group penalty.

    """

    def __init__(self, alpha: float, tau: float,
                 weights: np.ndarray, groups: np.ndarray):
        self.alpha, self.tau = alpha, tau
        self.weights, self.groups = weights, groups

    def value(self, w: np.ndarray) -> float:
        alpha, tau = self.alpha, self.tau
        weights, groups = self.weights, self.groups

        pen_l1_term = alpha * tau * norm(w, ord=1)

        pen_group_term = 0.
        for g in groups:
            if weights[g] == np.inf:
                continue
            pen_group_term += alpha * (1-tau) * weights[g] * norm(w[g], ord=2)

        return pen_l1_term + pen_group_term

    def prox_1feat(self, value: np.ndarray, stepsize: float, j: int) -> np.ndarray:
        tau, weight_j = self.tau, self.weights[j]
        mask_zero_coefs = value != 0

        # ST for each features
        value[mask_zero_coefs] = ST_vec(
            value[mask_zero_coefs],
            tau * stepsize
        )
        # bloc ST
        return BST(value, (1-tau) * stepsize * weight_j)

    def is_penalized(self, _):
        "Groups of features with inf weights are not penalized."
        n_feature_groups = [len(g) for g in self.groups]
        return np.repeat(self.weights, n_feature_groups) != np.inf

    def generalized_support(self, w):
        return w != 0

    def subdiff_distance(self, w, grad, ws):
        # still not implemented
        return
