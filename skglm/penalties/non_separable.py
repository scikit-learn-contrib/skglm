import numpy as np
from numba import float64

from skglm.penalties.base import BasePenalty
from skglm.utils.prox_funcs import prox_SLOPE


class SLOPE(BasePenalty):
    """Sorted L-One Penalized Estimation (SLOPE) penalty.

    Attributes
    ----------
    alphas : array, shape (n_features,)
        Contain regularization levels for every feature.
        When ``alphas`` contain a single unique value, ``SLOPE``
        is equivalent to the ``L1``penalty.

    References
    ----------
    .. [1] M. Bogdan, E. van den Berg, C. Sabatti, W. Su, E. Candes
        "SLOPE - Adaptive Variable Selection via Convex Optimization",
        The Annals of Applied Statistics 9 (3): 1103-40
        https://doi.org/10.1214/15-AOAS842
    """

    def __init__(self, alphas):
        self.alphas = alphas

    def get_spec(self):
        spec = (
            ('alphas', float64[:]),
        )
        return spec

    def params_to_dict(self):
        return dict(alphas=self.alphas)

    def value(self, w):
        """Compute the value of SLOPE at w."""
        return np.sum(np.sort(np.abs(w)) * self.alphas[::-1])

    def prox_vec(self, x, stepsize):
        alphas = self.alphas
        prox = np.zeros_like(x)

        abs_x = np.abs(x)
        sorted_indices = np.argsort(abs_x)[::-1]
        prox[sorted_indices] = prox_SLOPE(abs_x[sorted_indices], alphas * stepsize)

        return np.sign(x) * prox

    def subdiff_distance(self, w, grad, ws):
        raise ValueError(
            "No subdifferential distance for SLOPE. Use `opt_strategy='fixpoint'`"
        )
