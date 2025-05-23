import numpy as np
from numba import float64
from skglm.datafits.single_task import Huber
from skglm.utils.sparse_ops import spectral_norm


class QuantileHuber(Huber):
    r"""Quantile Huber loss for quantile regression.

    Implements the smoothed pinball loss with quadratic region:

    .. math::

       \rho_\tau^\delta(r) =
       \begin{cases}
           \tau\, r - \dfrac{\delta}{2}, & \text{if } r \ge \delta,\\
           \dfrac{\tau r^{2}}{2\delta}, & \text{if } 0 \le r < \delta,\\
           \dfrac{(1-\tau) r^{2}}{2\delta}, & \text{if } -\delta < r < 0,\\
           (\tau - 1)\, r - \dfrac{\delta}{2}, & \text{if } r \le -\delta.
       \end{cases}

    Parameters
    ----------
    quantile : float, default=0.5
        Desired quantile level between 0 and 1.
    delta : float, default=1.0
        Width of quadratic region.

    References
    ----------
    Chen, C. (2007). A Finite Smoothing Algorithm for Quantile Regression.
    Journal of Computational and Graphical Statistics, 16(1), 136–164.
    http://www.jstor.org/stable/27594233
    """

    def __init__(self, quantile=0.5, delta=1.0):
        if not 0 < quantile < 1:
            raise ValueError("quantile must be between 0 and 1")
        self.delta = float(delta)
        self.quantile = float(quantile)

    def get_spec(self):
        return (('delta', float64), ('quantile', float64))

    def params_to_dict(self):
        return dict(delta=self.delta, quantile=self.quantile)

    def _loss_and_grad_scalar(self, residual):
        """Calculate loss and gradient for a single residual."""
        tau = self.quantile
        delta = self.delta
        abs_r = abs(residual)

        # Quadratic core: |r| ≤ delta
        if abs_r <= delta:
            if residual >= 0:
                # 0 ≤ r ≤ delta
                loss = tau * residual**2 / (2 * delta)
                grad = tau * residual / delta
            else:
                # -delta ≤ r < 0
                loss = (1 - tau) * residual**2 / (2 * delta)
                grad = (1 - tau) * residual / delta
            return loss, grad

        # Linear tails: |r| > delta
        if residual > delta:
            loss = tau * (residual - delta / 2)
            grad = tau
            return loss, grad
        else:
            loss = (1 - tau) * (-residual - delta / 2)
            grad = tau - 1
            return loss, grad

    def value(self, y, w, Xw):
        """Compute the quantile Huber loss value."""
        residuals = y - Xw
        loss = np.zeros_like(residuals)
        for i, r in enumerate(residuals):
            loss[i], _ = self._loss_and_grad_scalar(r)
        return np.mean(loss)

    def raw_grad(self, y, Xw):
        """Compute gradient of datafit w.r.t Xw."""
        residuals = y - Xw
        grad = np.zeros_like(residuals)
        for i, r in enumerate(residuals):
            _, grad[i] = self._loss_and_grad_scalar(r)
        return -grad

    def get_lipschitz(self, X, y):
        """Compute coordinate-wise Lipschitz constants."""
        weight = max(self.quantile, 1 - self.quantile)
        return weight * (X ** 2).sum(axis=0) / (len(y) * self.delta)

    def get_global_lipschitz(self, X, y):
        """Compute global Lipschitz constant."""
        weight = max(self.quantile, 1 - self.quantile)
        return weight * np.linalg.norm(X, 2) ** 2 / (len(y) * self.delta)

    def get_lipschitz_sparse(self, X_data, X_indptr, X_indices, y):
        """Compute coordinate-wise Lipschitz constants for sparse X."""
        n_samples = len(y)
        weight = max(self.quantile, 1 - self.quantile)
        n_features = len(X_indptr) - 1
        lipschitz = np.zeros(n_features, dtype=X_data.dtype)
        for j in range(n_features):
            nrm2 = 0.0
            for idx in range(X_indptr[j], X_indptr[j + 1]):
                nrm2 += X_data[idx] ** 2
            lipschitz[j] = weight * nrm2 / (n_samples * self.delta)
        return lipschitz

    def get_global_lipschitz_sparse(self, X_data, X_indptr, X_indices, y):
        """Compute global Lipschitz constant for sparse X."""
        n_samples = len(y)
        weight = max(self.quantile, 1 - self.quantile)
        return weight * spectral_norm(
            X_data, X_indptr, X_indices, n_samples
        ) ** 2 / (n_samples * self.delta)

    def intercept_update_step(self, y, Xw):
        return -np.mean(self.raw_grad(y, Xw))
