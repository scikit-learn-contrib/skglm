import numpy as np
from numba import float64

from skglm.penalties.base import BasePenalty
from skglm.utils import ST


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
        is_w_ws_provided = len(w) == len(ws)

        for idx, j in enumerate(ws):
            w_j = w[idx] if is_w_ws_provided else w[j]
            if w_j == 0:
                # distance of - grad_j to  [-alpha, alpha]
                subdiff_dist[idx] = max(0, np.abs(grad[idx]) - self.alpha)
            else:
                # distance of - grad_j to alpha * sign(w[j])
                subdiff_dist[idx] = np.abs(
                    - grad[idx] - np.sign(w_j) * self.alpha)
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

    # used by PN-PAB
    def delta_pen(self, w_j, delta_w_j):
        delta_obj = 0.
        if w_j < 0:
            delta_obj -= self.alpha * delta_w_j
        elif w_j > 0:
            delta_obj += self.alpha * delta_w_j
        else:
            delta_obj -= self.alpha * abs(delta_w_j)
        return delta_obj
