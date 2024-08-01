import numpy as np
from numba import float64
from skglm.datafits import BaseDatafit
from skglm.utils.prox_funcs import ST_vec


class Pinball(BaseDatafit):
    r"""Pinball datafit.

    The datafit reads::

        sum_i quantile_level * max(y_i - Xw_i, 0) +
        (1 - quantile_level) * max(Xw_i - y_i, 0)

    with ``quantile_level`` in [0, 1].

    Parameters
    ----------
    quantile_level : float
        Quantile level must be in [0, 1]. When ``quantile_level=0.5``,
        the datafit becomes a Least Absolute Deviation (LAD) datafit.
    """

    def __init__(self, quantile_level):
        self.quantile_level = quantile_level

    def value(self, y, w, Xw):
        # implementation taken from
        # github.com/benchopt/benchmark_quantile_regression/blob/main/objective.py
        quantile_level = self.quantile_level

        residual = y - Xw
        sign = residual >= 0

        loss = (quantile_level * sign * residual -
                (1 - quantile_level) * (1 - sign) * residual)
        return np.sum(loss)

    def prox(self, w, step, y):
        """Prox of ``step * pinball``."""
        shift_cst = (self.quantile_level - 1/2) * step
        return y - ST_vec(y - w - shift_cst, step / 2)

    def prox_conjugate(self, z, step, y):
        """Prox of ``step * pinball^*``."""
        # using Moreau decomposition
        inv_step = 1 / step
        return z - step * self.prox(inv_step * z, inv_step, y)

    def subdiff_distance(self, Xw, z, y):
        """Distance of ``z`` to subdiff of pinball at ``Xw``."""
        # computation note: \partial ||y - . ||_1(Xw) = -\partial || . ||_1(y - Xw)
        y_minus_Xw = y - Xw
        shift_cst = self.quantile_level - 1/2

        max_distance = 0.
        for i in range(len(y)):

            if y_minus_Xw[i] == 0.:
                distance_i = max(0, abs(z[i] - shift_cst) - 1)
            else:
                distance_i = abs(z[i] + shift_cst + np.sign(y_minus_Xw[i]))

            max_distance = max(max_distance, distance_i)

        return max_distance

    def get_spec(self):
        spec = (
            ('quantile_level', float64),
        )
        return spec

    def params_to_dict(self):
        return dict(quantile_level=self.quantile_level)
