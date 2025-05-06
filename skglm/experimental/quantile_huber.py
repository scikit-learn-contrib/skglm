import numpy as np
from numpy.linalg import norm
from numba import float64
from skglm.datafits.base import BaseDatafit
from skglm.utils.sparse_ops import spectral_norm


class QuantileHuber(BaseDatafit):
    r"""Huber‑smoothed pinball loss for quantile regression.

    This class implements a smoothed approximation of the pinball (quantile)
    loss by applying Huber‑style smoothing at the non‑differentiable point.
    The formulation improves numerical stability and convergence for
    gradient‑based solvers, particularly on large data sets.

    Parameters
    ----------
    delta : float, positive
        Width of the quadratic region around the origin.  Larger values create
        stronger smoothing.  As ``delta`` -> 0, the loss approaches the
        standard pinball loss.

    quantile : float in (0, 1)
        Target quantile level (e.g. ``0.5`` corresponds to the median).

    Notes
    -----
    The loss function is defined as

    .. math::

        L(r) =
        \\begin{cases}
            \\tau \\dfrac{r^{2}}{2\\delta}, & 0 < r \\le \\delta \\\\
            (1-\\tau) \\dfrac{r^{2}}{2\\delta}, & -\\delta \\le r < 0 \\\\
            \\tau \\left(r - \\dfrac{\\delta}{2}\\right), & r > \\delta \\\\
            (1-\\tau) \\left(-r - \\dfrac{\\delta}{2}\\right), & r < -\\delta
        \\end{cases}

    where :math:`r = y - Xw` is the residual, :math:`\\tau` is the target
    quantile, and :math:`\\delta` controls the smoothing width.

    References
    ----------
    He, X., Pan, X., Tan, K. M., & Zhou, W. X. (2021).
    *Smoothed Quantile Regression with Large‑Scale Inference*.
    """

    def __init__(self, delta, quantile):
        if not 0 < quantile < 1:
            raise ValueError("quantile must be between 0 and 1")
        if delta <= 0:
            raise ValueError("delta must be positive")
        self.delta = float(delta)
        self.quantile = float(quantile)

    def get_spec(self):
        spec = (
            ('delta', float64),
            ('quantile', float64),
        )
        return spec

    def params_to_dict(self):
        return dict(delta=self.delta, quantile=self.quantile)

    def get_lipschitz(self, X, y):
        n_samples = len(y)
        weight = max(self.quantile, 1 - self.quantile)

        lipschitz = weight * (X ** 2).sum(axis=0) / (n_samples * self.delta)
        return lipschitz

    def get_lipschitz_sparse(self, X_data, X_indptr, X_indices, y):
        n_samples = len(y)
        n_features = len(X_indptr) - 1
        weight = max(self.quantile, 1 - self.quantile)

        lipschitz = np.zeros(n_features, dtype=X_data.dtype)
        for j in range(n_features):
            nrm2 = 0.0
            for idx in range(X_indptr[j], X_indptr[j + 1]):
                nrm2 += X_data[idx] ** 2
            lipschitz[j] = weight * nrm2 / (n_samples * self.delta)
        return lipschitz

    def get_global_lipschitz(self, X, y):
        n_samples = len(y)
        weight = max(self.quantile, 1 - self.quantile)
        return weight * norm(X, 2) ** 2 / (n_samples * self.delta)

    def get_global_lipschitz_sparse(self, X_data, X_indptr, X_indices, y):
        n_samples = len(y)
        weight = max(self.quantile, 1 - self.quantile)
        return (
            weight
            * spectral_norm(X_data, X_indptr, X_indices, n_samples) ** 2
            / (n_samples * self.delta)
        )

    def _loss_and_grad_scalar(self, residual):
        tau, delta = self.quantile, self.delta
        abs_r = abs(residual)

        if abs_r <= delta:
            # Quadratic region
            if residual > 0:
                return tau * residual**2 / (2 * delta), tau * residual / delta
            else:
                return ((1 - tau) * residual**2 / (2 * delta), (1 - tau)
                        * residual / delta
                        )

        # Linear tails
        if residual > delta:
            return tau * (residual - delta/2), tau
        else:  # residual < -delta
            return (1 - tau) * (-residual - delta/2), -(1 - tau)

    def value(self, y, w, Xw):
        n_samples = len(y)
        res = 0.0
        for i in range(n_samples):
            loss_i, _ = self._loss_and_grad_scalar(y[i] - Xw[i])
            res += loss_i
        return res / n_samples

    def _dr(self, residual):
        """Compute dl/dr for each residual."""
        tau = self.quantile
        delt = self.delta

        # Pick tau for r >= 0, (1 - tau) for r < 0
        scale = np.where(residual >= 0, tau, 1 - tau)

        # Inside the quadratic zone: slope = scale * (r / delt)
        # Outside: slope is ± scale, same sign as r
        dr = np.where(
            np.abs(residual) <= delt,
            scale * (residual / delt),
            np.sign(residual) * scale
        )
        return dr

    def gradient_scalar(self, X, y, w, Xw, j):
        r = y - Xw
        dr = self._dr(r)
        return - X[:, j].dot(dr) / len(y)

    def gradient_scalar_sparse(self, X_data, X_indptr, X_indices, y, Xw, j):
        r = y - Xw
        dr = self._dr(r)
        idx_start, idx_end = X_indptr[j], X_indptr[j + 1]
        rows = X_indices[idx_start:idx_end]
        vals = X_data[idx_start:idx_end]
        return - np.dot(vals, dr[rows]) / len(y)

    def full_grad_sparse(self, X_data, X_indptr, X_indices, y, Xw):
        n_features = len(X_indptr) - 1
        n_samples = len(y)
        grad = np.zeros(n_features, dtype=Xw.dtype)
        for j in range(n_features):
            g = 0.0
            for idx in range(X_indptr[j], X_indptr[j + 1]):
                i = X_indices[idx]
                residual = y[i] - Xw[i]
                _, grad_r = self._loss_and_grad_scalar(residual)
                g += -X_data[idx] * grad_r
            grad[j] = g / n_samples
        return grad

    def intercept_update_step(self, y, Xw):
        n_samples = len(y)
        update = 0.0
        for i in range(n_samples):
            residual = y[i] - Xw[i]
            _, grad_r = self._loss_and_grad_scalar(residual)
            update += -grad_r
        return update / n_samples

    def initialize(self, X, y):
        pass

    def initialize_sparse(self, X_data, X_indptr, X_indices, y):
        pass

    def gradient(self, X, y, Xw):
        n_samples, n_features = X.shape
        grad = np.zeros(n_features)
        for j in range(n_features):
            grad[j] = self.gradient_scalar(X, y, None, Xw, j)
        return grad
