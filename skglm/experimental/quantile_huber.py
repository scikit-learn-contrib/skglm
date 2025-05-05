import numpy as np
from numpy.linalg import norm
from numba import float64
from skglm.datafits.base import BaseDatafit
from skglm.utils.sparse_ops import spectral_norm


class QuantileHuber(BaseDatafit):
    """Huber-smoothed Pinball (Quantile) loss.

    This loss function is a smoothed version of the Pinball loss used for quantile
    regression. By smoothing the non-differentiable point with a quadratic region,
    it improves numerical stability for solvers like PDCD_WS on larger datasets.

    Parameters
    ----------
    delta : float, positive
        Width of the quadratic region around the origin. Larger values create
        more smoothing. As delta approaches 0, this approaches the standard Pinball loss.

    quantile : float, between 0 and 1
        The desired quantile level. For example, 0.5 corresponds to the median.

    Notes
    -----
    For a residual r = y - Xw, the loss is defined as:

        For |r| ≤ delta:
            loss = (τ if r > 0 else (1-τ)) * (r²/(2*delta))

        For r > delta:
            loss = τ * (r - delta/2)

        For r < -delta:
            loss = (1-τ) * (-r - delta/2)

    Where τ is the quantile parameter.

    * When τ = 0.5, this reduces to the symmetric Huber loss used for median regression.
    * As delta → 0, it converges to the standard Pinball loss.

    References
    ----------
    Friedman, J.H., "Greedy Function Approximation: A Gradient Boosting Machine,"
    Annals of Statistics, 29(5):1189-1232, 2001.
    """

    def __init__(self, delta, quantile):
        if not 0 < quantile < 1:
            raise ValueError("quantile must be between 0 and 1")
        if delta <= 0:
            raise ValueError("delta must be positive")
        self.delta = float(delta)
        self.quantile = float(quantile)

    def get_spec(self):
        return (
            ('delta', float64),
            ('quantile', float64),
        )

    def params_to_dict(self):
        return dict(delta=self.delta, quantile=self.quantile)

    def get_lipschitz(self, X, y):
        """Return coordinatewise Lipschitz constants of the gradient.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values.

        Returns
        -------
        lipschitz : array, shape (n_features,)
            Coordinatewise Lipschitz constants.
        """
        n_samples = len(y)
        weight = max(self.quantile, 1 - self.quantile)
        lipschitz = weight * (X ** 2).sum(axis=0) / (n_samples * self.delta)
        return lipschitz

    def get_lipschitz_sparse(self, X_data, X_indptr, X_indices, y):
        """Return coordinatewise Lipschitz constants for sparse data."""
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
        """Return global Lipschitz constant of the gradient."""
        n_samples = len(y)
        return norm(X, 2) ** 2 / (n_samples * self.delta)

    def get_global_lipschitz_sparse(self, X_data, X_indptr, X_indices, y):
        """Return global Lipschitz constant for sparse data."""
        n_samples = len(y)
        return spectral_norm(X_data, X_indptr, X_indices, n_samples) ** 2 / (n_samples * self.delta)

    def _loss_and_grad_scalar(self, residual):
        """Return (loss, dℓ/dr) for one residual value."""
        tau, delta = self.quantile, self.delta
        abs_r = abs(residual)

        if abs_r <= delta:
            # Quadratic region
            if residual > 0:
                return tau * residual**2 / (2 * delta), tau * residual / delta
            else:
                return (1 - tau) * residual**2 / (2 * delta), (1 - tau) * residual / delta

        # Linear tails
        if residual > delta:
            return tau * (residual - delta/2), tau
        else:  # residual < -delta
            return (1 - tau) * (-residual - delta/2), -(1 - tau)

    def value(self, y, w, Xw):
        """Compute the value of the loss.

        Parameters
        ----------
        y : array-like, shape (n_samples,)
            Target values.
        w : array-like, shape (n_features,)
            Coefficient vector.
        Xw : array-like, shape (n_samples,)
            Model predictions.

        Returns
        -------
        float
            Value of the loss.
        """
        n_samples = len(y)
        res = 0.0
        for i in range(n_samples):
            loss_i, _ = self._loss_and_grad_scalar(y[i] - Xw[i])
            res += loss_i
        return res / n_samples

    def _dr(self, residual):
        # residual: array of shape (n_samples,)
        τ, δ = self.quantile, self.delta
        # scale = τ if r>=0 else (1-τ)
        scale = np.where(residual >= 0, τ, 1-τ)
        # in the quadratic region, slope = scale * (r/δ), else ±scale
        dr = np.where(
            np.abs(residual) <= δ,
            scale * (residual / δ),
            np.sign(residual) * scale
        )
        return dr

    def gradient_scalar(self, X, y, w, Xw, j):
        # 1) compute dr once
        r = y - Xw
        dr = self._dr(r)
        # 2) dot‐product for feature j
        #    equivalent to sum_i X[i,j] * dr[i]
        return - X[:, j].dot(dr) / len(y)

    def gradient_scalar_sparse(self, X_data, X_indptr, X_indices, y, Xw, j):
        # 1) compute dr once (dense vector)
        r = y - Xw
        dr = self._dr(r)
        # 2) only loop over nonzero entries of column j,
        #    but use the precomputed dr values
        idx_start, idx_end = X_indptr[j], X_indptr[j + 1]
        rows = X_indices[idx_start:idx_end]        # which sample each entry belongs to
        vals = X_data[idx_start:idx_end]           # the nonzero values
        # grad_j = - sum_k vals[k] * dr[ rows[k] ]
        return - np.dot(vals, dr[rows]) / len(y)

    def full_grad_sparse(self, X_data, X_indptr, X_indices, y, Xw):
        """Compute the full gradient with sparse data."""
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
        """Compute the update step for the intercept."""
        n_samples = len(y)
        update = 0.0
        for i in range(n_samples):
            residual = y[i] - Xw[i]
            _, grad_r = self._loss_and_grad_scalar(residual)
            update += -grad_r
        return update / n_samples

    def initialize(self, X, y):
        """Precompute constants to speed up computations.
        This is a no-op for this loss function.
        """
        pass

    def initialize_sparse(self, X_data, X_indptr, X_indices, y):
        """Precompute constants for sparse data to speed up computations.
        This is a no-op for this loss function.
        """
        pass

    def gradient(self, X, y, Xw):
        """Compute the gradient of the datafit.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Design matrix.
        y : array-like, shape (n_samples,)
            Target vector.
        Xw : array-like, shape (n_samples,)
            Model predictions.

        Returns
        -------
        grad : array, shape (n_features,)
            Gradient vector.
        """
        n_samples, n_features = X.shape
        grad = np.zeros(n_features)
        for j in range(n_features):
            grad[j] = self.gradient_scalar(X, y, None, Xw, j)
        return grad
