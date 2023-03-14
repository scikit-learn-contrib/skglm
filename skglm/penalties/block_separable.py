import numpy as np
from numpy.linalg import norm

from numba import float64, int32

from skglm.penalties.base import BasePenalty
from skglm.utils.prox_funcs import (
    BST, prox_block_2_05, prox_SCAD, value_SCAD, prox_MCP, value_MCP)


class L2_1(BasePenalty):
    """L2/1 row-wise penalty: sum of L2 norms of rows."""

    def __init__(self, alpha):
        self.alpha = alpha

    def get_spec(self):
        spec = (
            ('alpha', float64),
        )
        return spec

    def params_to_dict(self):
        return dict(alpha=self.alpha)

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
        return np.ones(n_features, dtype=np.bool_)


class L2_05(BasePenalty):
    """L2/0.5 row-wise penalty: sum of square roots of L2 norms of rows."""

    def __init__(self, alpha):
        self.alpha = alpha

    def get_spec(self):
        spec = (
            ('alpha', float64),
        )
        return spec

    def params_to_dict(self):
        return dict(alpha=self.alpha)

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
        return np.ones(n_features, dtype=np.bool_)


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

    def get_spec(self):
        spec = (
            ('alpha', float64),
            ('gamma', float64),
        )
        return spec

    def params_to_dict(self):
        return dict(alpha=self.alpha,
                    gamma=self.gamma)

    def value(self, W):
        """Compute the value of BlockMCP at W."""
        norm_rows = np.sqrt(np.sum(W ** 2, axis=1))
        return value_MCP(norm_rows, self.alpha, self.gamma)

    def prox_1feat(self, value, stepsize, j):
        """Compute the proximal operator of BlockMCP."""
        norm_rows = norm(value)
        prox = prox_MCP(norm_rows, stepsize, self.alpha, self.gamma)
        return prox * value / norm_rows

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
        return np.ones(n_features, dtype=np.bool_)


class BlockSCAD(BasePenalty):
    r"""Block Smoothly Clipped Absolute Deviation.

    Notes
    -----
    With :math:`W_j` the j-th row of W, the penalty is:

    .. math::
        "pen"(||W_j||) = {
            (alpha ||W_j||            , if \ \ \ \ \ \ \ \ \ \ ||W_j|| <= alpha),
            ((2 alpha gamma ||W_j|| - ||W_j||^2 - alpha^2) / (2 (gamma - 1))
                                      , if       alpha  \ \  < ||W_j|| <= alpha gamma),
            ((alpha^2 (gamma + 1)) / 2, if       alpha gamma < ||W_j||)
        :}

    .. math::
        "value" = sum_(j=1)^(n_"features") "pen"(||W_j||)
    """

    def __init__(self, alpha, gamma):
        self.alpha = alpha
        self.gamma = gamma

    def get_spec(self):
        spec = (
            ('alpha', float64),
            ('gamma', float64),
        )
        return spec

    def params_to_dict(self):
        return dict(alpha=self.alpha,
                    gamma=self.gamma)

    def value(self, W):
        """Compute the value of the SCAD penalty at W."""
        norm_rows = np.sqrt(np.sum(W ** 2, axis=1))
        return value_SCAD(norm_rows, self.alpha, self.gamma)

    def prox_1feat(self, value, stepsize, j):
        """Compute the proximal operator of BlockSCAD."""
        norm_value = norm(value)
        prox = prox_SCAD(norm_value, stepsize, self.alpha, self.gamma)
        return prox * value / norm_value

    def subdiff_distance(self, W, grad, ws):
        """Compute distance of negative gradient to the subdifferential at W."""
        subdiff_dist = np.zeros_like(ws, dtype=grad.dtype)
        for idx, j in enumerate(ws):
            norm_Wj = norm(W[j])
            if not np.any(W[j]):
                # distance of -grad_j to alpha * unit_ball
                subdiff_dist[idx] = max(0, norm(grad[idx]) - self.alpha)
            elif norm_Wj <= self.alpha:
                # distance of -grad_j to alpha * W[j] / ||W[j]||
                subdiff_dist[idx] = norm(grad[idx] + self.alpha * W[j] / norm_Wj)
            elif norm_Wj <= self.gamma * self.alpha:
                # distance of -grad_j to (alpha * gamma - ||W[j]||)
                # / ((gamma - 1) * ||W[j]||) * W[j]
                subdiff_dist[idx] = norm(grad[idx] + (
                    (self.alpha * self.gamma - norm_Wj) / (norm_Wj * (self.gamma - 1))
                ) * W[j])
            else:
                # distance of -grad_j to 0
                subdiff_dist[idx] = norm(grad[idx])
        return subdiff_dist

    def is_penalized(self, n_features):
        """Return a binary mask with the penalized features."""
        return np.ones(n_features, dtype=np.bool_)


class WeightedGroupL2(BasePenalty):
    r"""Weighted Group L2 penalty.

    The penalty reads

    .. math::
        sum_{g=1}^{n_"groups"} "weights"_g xx ||w_{[g]}||

    with :math:`w_{[g]}` being the coefficients of the g-th group.

    Attributes
    ----------
    alpha : float
        The regularization parameter.

    weights : array, shape (n_groups,)
        The weights of the groups.

    grp_indices : array, shape (n_features,)
        The group indices stacked contiguously
        ([grp1_indices, grp2_indices, ...]).

    grp_ptr : array, shape (n_groups + 1,)
        The group pointers such that two consecutive elements delimit
        the indices of a group in ``grp_indices``.
    """

    def __init__(self, alpha, weights, grp_ptr, grp_indices):
        self.alpha, self.weights = alpha, weights
        self.grp_ptr, self.grp_indices = grp_ptr, grp_indices

    def get_spec(self):
        spec = (
            ('alpha', float64),
            ('weights', float64[:]),
            ('grp_ptr', int32[:]),
            ('grp_indices', int32[:]),
        )
        return spec

    def params_to_dict(self):
        return dict(alpha=self.alpha, weights=self.weights,
                    grp_ptr=self.grp_ptr, grp_indices=self.grp_indices)

    def value(self, w):
        """Value of penalty at vector ``w``."""
        alpha, weights = self.alpha, self.weights
        grp_ptr, grp_indices = self.grp_ptr, self.grp_indices
        n_grp = len(grp_ptr) - 1

        sum_weighted_L2 = 0.
        for g in range(n_grp):
            grp_g_indices = grp_indices[grp_ptr[g]: grp_ptr[g+1]]
            w_g = w[grp_g_indices]

            sum_weighted_L2 += alpha * weights[g] * norm(w_g)

        return sum_weighted_L2

    def prox_1group(self, value, stepsize, g):
        """Compute the proximal operator of group ``g``."""
        return BST(value, self.alpha * stepsize * self.weights[g])

    def subdiff_distance(self, w, grad_ws, ws):
        """Compute distance to the subdifferential at ``w`` of negative gradient.

        Note: ``grad_ws`` is a stacked array of gradients.
        ([grad_ws_1, grad_ws_2, ...])
        """
        alpha, weights = self.alpha, self.weights
        grp_ptr, grp_indices = self.grp_ptr, self.grp_indices

        scores = np.zeros(len(ws))
        grad_ptr = 0
        for idx, g in enumerate(ws):
            grp_g_indices = grp_indices[grp_ptr[g]: grp_ptr[g+1]]

            grad_g = grad_ws[grad_ptr: grad_ptr + len(grp_g_indices)]
            grad_ptr += len(grp_g_indices)

            w_g = w[grp_g_indices]
            norm_w_g = norm(w_g)

            if norm_w_g == 0:
                scores[idx] = max(0, norm(grad_g) - alpha * weights[g])
            else:
                subdiff = alpha * weights[g] * w_g / norm_w_g
                scores[idx] = norm(grad_g + subdiff)

        return scores

    def is_penalized(self, n_groups):
        return np.ones(n_groups, dtype=np.bool_)

    def generalized_support(self, w):
        grp_indices, grp_ptr = self.grp_indices, self.grp_ptr
        n_groups = len(grp_ptr) - 1
        is_penalized = self.is_penalized(n_groups)

        gsupp = np.zeros(n_groups, dtype=np.bool_)
        for g in range(n_groups):
            if not is_penalized[g]:
                gsupp[g] = True
                continue

            grp_g_indices = grp_indices[grp_ptr[g]: grp_ptr[g+1]]
            if np.any(w[grp_g_indices]):
                gsupp[g] = True

        return gsupp
