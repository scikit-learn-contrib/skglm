import numpy as np
from numba import float64
from numba.types import bool_

from skglm.penalties.base import BasePenalty
from skglm.utils import (
    ST, box_proj, prox_05, prox_2_3, prox_SCAD, value_SCAD, prox_MCP, value_MCP)


class L1(BasePenalty):
    """L1 penalty."""

    def __init__(self, alpha):
        self.alpha = alpha

    def get_spec(self):
        spec = (
            ('alpha', float64),
        )
        return spec

    def params_to_dict(self):
        return dict(alpha=self.alpha)

    def value(self, w):
        """Compute L1 penalty value."""
        return self.alpha * np.sum(np.abs(w))

    def prox_1d(self, value, stepsize, j):
        """Compute proximal operator of the L1 penalty (soft-thresholding operator)."""
        return ST(value, self.alpha * stepsize)

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
        return subdiff_dist

    def is_penalized(self, n_features):
        """Return a binary mask with the penalized features."""
        return np.ones(n_features, dtype=np.bool_)

    def generalized_support(self, w):
        """Return a mask with non-zero coefficients."""
        return w != 0

    def alpha_max(self, gradient0):
        """Return penalization value for which 0 is solution."""
        return np.max(np.abs(gradient0))


class L1_plus_L2(BasePenalty):
    """L1 + L2 penalty (aka ElasticNet penalty)."""

    def __init__(self, alpha, l1_ratio):
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def get_spec(self):
        spec = (
            ('alpha', float64),
            ('l1_ratio', float64),
        )
        return spec

    def params_to_dict(self):
        return dict(alpha=self.alpha,
                    l1_ratio=self.l1_ratio)

    def value(self, w):
        """Compute the L1 + L2 penalty value."""
        value = self.l1_ratio * self.alpha * np.sum(np.abs(w))
        value += (1 - self.l1_ratio) * self.alpha / 2 * np.sum(w ** 2)
        return value

    def prox_1d(self, value, stepsize, j):
        """Compute the proximal operator (scaled soft-thresholding)."""
        prox = ST(value, self.l1_ratio * self.alpha * stepsize)
        prox /= (1 + stepsize * (1 - self.l1_ratio) * self.alpha)
        return prox

    def subdiff_distance(self, w, grad, ws):
        """Compute distance of negative gradient to the subdifferential at w."""
        subdiff_dist = np.zeros_like(grad)
        for idx, j in enumerate(ws):
            if w[j] == 0:
                # distance of - grad_j to alpha * l1_ratio * [-1, 1]
                subdiff_dist[idx] = max(
                    0, np.abs(grad[idx]) - self.alpha * self.l1_ratio)
            else:
                # distance of - grad_j to alpha * l_1 ratio * sign(w[j]) +
                # alpha * (1 - l1_ratio) * w[j]
                subdiff_dist[idx] = np.abs(
                    - grad[idx] -
                    self.alpha * (self.l1_ratio *
                                  np.sign(w[j]) + (1 - self.l1_ratio) * w[j]))
        return subdiff_dist

    def is_penalized(self, n_features):
        """Return a binary mask with the penalized features."""
        return np.ones(n_features).astype(bool_)

    def generalized_support(self, w):
        """Return a mask with non-zero coefficients."""
        return w != 0

    def alpha_max(self, gradient0):
        """Return penalization value for which 0 is solution."""
        return np.max(np.abs(gradient0))


class WeightedL1(BasePenalty):
    """Weighted L1 penalty."""

    def __init__(self, alpha, weights):
        self.alpha = alpha
        self.weights = weights.astype(np.float64)

    def get_spec(self):
        spec = (
            ('alpha', float64),
            ('weights', float64[:]),
        )
        return spec

    def params_to_dict(self):
        return dict(alpha=self.alpha,
                    weights=self.weights)

    def value(self, w):
        """Compute the weighted L1 penalty."""
        return self.alpha * np.sum(np.abs(w) * self.weights)

    def prox_1d(self, value, stepsize, j):
        """Compute the proximal operator of weighted L1 (weighted soft-thresholding)."""
        return ST(value, self.alpha * stepsize * self.weights[j])

    def subdiff_distance(self, w, grad, ws):
        """Compute distance of negative gradient to the subdifferential at w."""
        subdiff_dist = np.zeros_like(grad)
        for idx, j in enumerate(ws):
            if w[j] == 0:
                # distance of - grad_j to alpha * weights[j] * [-1, 1]
                subdiff_dist[idx] = max(
                    0, np.abs(grad[idx]) - self.alpha * self.weights[j])
            else:
                # distance of - grad_j to alpha * weights[j] * sign(w[j])
                subdiff_dist[idx] = np.abs(
                    - grad[idx] - self.alpha * self.weights[j] * np.sign(w[j]))
        return subdiff_dist

    def is_penalized(self, n_features):
        """Return a binary mask with the penalized features."""
        return self.weights != 0

    def generalized_support(self, w):
        """Return a mask with non-zero coefficients."""
        return w != 0

    def alpha_max(self, gradient0):
        """Return penalization value for which 0 is solution."""
        nnz_weights = self.weights != 0
        return np.max(np.abs(gradient0[nnz_weights] / self.weights[nnz_weights]))


class MCPenalty(BasePenalty):
    """Minimax Concave Penalty (MCP), a non-convex sparse penalty.

    Notes
    -----
    With x >= 0
    pen(x) =
    alpha * x - x^2 / (2 * gamma) if x =< gamma * alpha
    gamma * alpha^2 / 2           if x > gamma * alpha
    value = sum_{j=1}^{n_features} pen(abs(w_j))
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

    def value(self, w):
        return value_MCP(w, self.alpha, self.gamma)

    def prox_1d(self, value, stepsize, j):
        """Compute the proximal operator of MCP."""
        return prox_MCP(value, stepsize, self.alpha, self.gamma)

    def subdiff_distance(self, w, grad, ws):
        """Compute distance of negative gradient to the subdifferential at w."""
        subdiff_dist = np.zeros_like(grad)
        for idx, j in enumerate(ws):
            if w[j] == 0:
                # distance of -grad to alpha * [-1, 1]
                subdiff_dist[idx] = max(0, np.abs(grad[idx]) - self.alpha)
            elif np.abs(w[j]) < self.alpha * self.gamma:
                # distance of -grad_j to (alpha * sign(w[j]) - w[j] / gamma)
                subdiff_dist[idx] = np.abs(
                    grad[idx] + self.alpha * np.sign(w[j]) - w[j] / self.gamma)
            else:
                # distance of grad to 0
                subdiff_dist[idx] = np.abs(grad[idx])
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


class SCAD(BasePenalty):
    """Smoothly Clipped Absolute Deviation.

    Notes
    -----
    With x >= 0
    pen(x) =
    alpha * x                         if x =< alpha
    2 * gamma * alpha * x - x^2 - alpha^2 \
        / 2 * (gamma - 1))            if alpha < x < alpha * gamma
    alpha^2 * (gamma + 1) / 2      if x > gamma * alpha
    value = sum_{j=1}^{n_features} pen(abs(w_j))
    """

    def __init__(self, alpha, gamma):
        self.alpha = alpha
        self.gamma = gamma

    def get_spec(self):
        spec = (
            ('alpha', float64),
            ('gamma', float64)
        )
        return spec

    def params_to_dict(self):
        return dict(alpha=self.alpha,
                    gamma=self.gamma)

    def value(self, w):
        """Compute the value of the SCAD penalty at w."""
        return value_SCAD(w, self.alpha, self.gamma)

    def prox_1d(self, value, stepsize, j):
        """Compute the proximal operator of SCAD penalty."""
        return prox_SCAD(value, stepsize, self.alpha, self.gamma)

    def subdiff_distance(self, w, grad, ws):
        """Compute distance of negative gradient to the subdifferential at w."""
        subdiff_dist = np.zeros_like(grad)
        for idx, j in enumerate(ws):
            if w[j] == 0:
                # distance of -grad_j to alpha * [-1, 1]
                subdiff_dist[idx] = max(0, np.abs(grad[idx]) - self.alpha)
            elif np.abs(w[j]) <= self.alpha:
                # distance of -grad_j to alpha * sgn(w[j])
                subdiff_dist[idx] = np.abs(grad[idx] + self.alpha * np.sign(w[j]))
            elif np.abs(w[j]) <= self.alpha * self.gamma:
                # distance of -grad_j to (alpha * gamma * sign(w[j]) - w[j])
                #                        / (gamma - 1)
                subdiff_dist[idx] = np.abs(
                    grad[idx] +
                    (np.sign(w[j]) * self.alpha * self.gamma - w[j]) / (self.gamma - 1)
                )
            else:
                # distance of -grad_j to 0
                subdiff_dist[idx] = np.abs(grad[idx])
        return subdiff_dist

    def is_penalized(self, n_features):
        """Return a binary mask with the penalized features."""
        return np.ones(n_features, bool_)

    def generalized_support(self, w):
        """Return a mask with non-zero coefficients."""
        return w != 0


class IndicatorBox(BasePenalty):
    """Box constraint penalty.

    Notes
    -----
    ind_[0, alpha]^n_samples
    where ind is the indicator function of the convex set
    [0, alpha]^n_samples
    """

    def __init__(self, alpha):
        self.alpha = alpha

    def get_spec(self):
        spec = (
            ('alpha', float64),
        )
        return spec

    def params_to_dict(self):
        return dict(alpha=self.alpha)

    def value(self, w):
        """Compute the value of the IndicatorBox at w."""
        if np.max(w) > self.alpha:
            return np.inf
        elif np.min(w) < 0:
            return np.inf
        return 0.0

    def prox_1d(self, value, stepsize, j):
        """Compute the proximal operator of the Indicator Box (box projection)."""
        return box_proj(value, 0, self.alpha)

    def subdiff_distance(self, w, grad, ws):
        """Compute distance of negative gradient to the subdifferential at w."""
        subdiff_dist = np.zeros_like(grad)
        for idx, j in enumerate(ws):
            if w[j] == 0:
                # distance of - grad_j to  [-infty, 0]
                subdiff_dist[idx] = max(0, - grad[idx])
            elif w[j] == self.alpha:
                # distance of - grad_j to  [0, +infty]
                subdiff_dist[idx] = max(0, grad[idx])
            else:
                # distance of - grad_j to 0
                subdiff_dist[idx] = np.abs(grad[idx])
        return subdiff_dist

    def is_penalized(self, n_features):
        """Return a binary mask with the penalized features."""
        return np.ones(n_features, bool_)

    def generalized_support(self, w):
        """Return a mask with coefficients that are neither 0 nor alpha."""
        # w is the output of the projection unto [0, C] so checking strict equality
        # should be ok and we can avoid np.isclose
        return np.logical_and(w != 0, w != self.alpha)


class L0_5(BasePenalty):
    """L_{0.5} non-convex quasi-norm penalty."""

    def __init__(self, alpha):
        self.alpha = alpha

    def get_spec(self):
        spec = (
            ('alpha', float64),
        )
        return spec

    def params_to_dict(self):
        return dict(alpha=self.alpha)

    def value(self, w):
        """Compute the value of L0_5 at w."""
        return self.alpha * np.sum(np.abs(w) ** 0.5)

    def derivative(self, w):
        """Compute the element-wise derivative."""
        return 1. / (2. * np.sqrt(np.abs(w)) + 1e-12)

    def prox_1d(self, value, stepsize, j):
        """Compute the proximal operator of L0_5."""
        return prox_05(value, self.alpha * stepsize)

    def subdiff_distance(self, w, grad, ws):
        """Compute distance of negative gradient to the subdifferential at w."""
        subdiff_dist = np.zeros_like(grad)
        for idx, j in enumerate(ws):
            # tau = (3./2.) * (self.alpha / lc[j]) ** (2./3.)
            if w[j] == 0:
                # distance of - grad_j to  [-tau, tau]
                # subdiff_dist[idx] = max(0, np.abs(grad[idx]) / lc[j] - tau)
                subdiff_dist[idx] = 0.
            else:
                # distance of - grad_j to alpha * sign(w[j]) TODO fix comment
                subdiff_dist[idx] = np.abs(
                    - grad[idx] - np.sign(w[j]) *
                    self.alpha / (2 * np.sqrt(np.abs(w[j]))))

        return subdiff_dist

    def is_penalized(self, n_features):
        """Return a binary mask with the penalized features."""
        return np.ones(n_features, bool_)

    def generalized_support(self, w):
        """Return a mask with non-zero coefficients."""
        return w != 0


class L2_3(BasePenalty):
    """L_{2/3} quasi-norm non-convex penalty."""

    def __init__(self, alpha):
        self.alpha = alpha

    def get_spec(self):
        spec = (
            ('alpha', float64),
        )
        return spec

    def params_to_dict(self):
        return dict(alpha=self.alpha)

    def value(self, w):
        """Compute the value of the L2_3 norm at w."""
        return self.alpha * np.sum(np.abs(w) ** (2/3))

    def derivative(self, w):
        """Compute the element-wise derivative."""
        return 2 / (3 * np.abs(w) ** (1/3) + 1e-12)

    def prox_1d(self, value, stepsize, j):
        """Compute the proximal operator of the L2_3 norm."""
        return prox_2_3(value, self.alpha * stepsize)

    def subdiff_distance(self, w, grad, ws):
        """Compute distance of negative gradient to the subdifferential at w."""
        subdiff_dist = np.zeros_like(grad)
        for idx, j in enumerate(ws):
            # tau = 2. * (2./3. * self.alpha / lc[j]) ** (3./4.)

            if w[j] == 0:
                # distance of - grad_j to  [-tau, tau]
                # subdiff_dist[idx] = max(0, np.abs(grad[idx]) / lc[j] - tau)
                subdiff_dist[idx] = 0.
            else:
                # distance of - grad_j to alpha * sign(w[j]) TODO fix comment
                subdiff_dist[idx] = np.abs(
                    - grad[idx] - np.sign(w[j]) *
                    self.alpha * 2 / (3 * np.abs(w[j]) ** (1/3)))

        return subdiff_dist

    def is_penalized(self, n_features):
        """Return a binary mask with the penalized features."""
        return np.ones(n_features, bool_)

    def generalized_support(self, w):
        """Return a mask with non-zero coefficients."""
        return w != 0


class PositiveConstraint(BasePenalty):
    """Non-negative least square penalty."""

    def get_spec(self):
        return ()

    def params_to_dict(self):
        return dict()

    def value(self, w):
        """Compute the value of the PositiveConstraint penalty at w."""
        return np.inf if (w < 0).any() else 0.

    def prox_1d(self, value, stepsize, j):
        """Compute the proximal operator of the PositiveConstraint."""
        return max(0., value)

    def subdiff_distance(self, w, grad, ws):
        """Compute distance of negative gradient to the subdifferential at w."""
        subdiff_dist = np.zeros_like(grad)
        for idx, j in enumerate(ws):
            if w[j] == 0:
                # distance of - grad_j to  ]-infty, 0]
                subdiff_dist[idx] = max(0, -grad[idx])
            elif w[j] > 0:
                # distance of - grad_j to 0
                subdiff_dist[idx] = abs(-grad[idx])
            else:
                # subdiff is empty, distance is infinite
                subdiff_dist[idx] = np.inf

        return subdiff_dist

    def is_penalized(self, n_features):
        """Return a binary mask with the penalized features."""
        return np.ones(n_features, bool_)

    def generalized_support(self, w):
        """Return a mask with non-zero coefficients."""
        return w != 0
