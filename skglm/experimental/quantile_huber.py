import numpy as np
from numpy.linalg import norm
from numba import float64

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError

from skglm.datafits.base import BaseDatafit
from skglm.solvers import AndersonCD
from skglm.penalties import L1
from skglm.estimators import GeneralizedLinearEstimator
from skglm.utils.sparse_ops import spectral_norm


class QuantileHuber(BaseDatafit):
    r"""Quantile Huber loss for quantile regression.

    Implements the smoothed pinball loss:

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
        Smoothing parameter (0 mean no smoothing).
    """

    def __init__(self, quantile=0.5, delta=1.0):
        if not 0 < quantile < 1:
            raise ValueError("quantile must be between 0 and 1")
        if delta <= 0:
            raise ValueError("delta must be positive")
        self.delta = float(delta)
        self.quantile = float(quantile)

    def get_spec(self):
        return (('delta', float64), ('quantile', float64))

    def params_to_dict(self):
        return dict(delta=self.delta, quantile=self.quantile)

    def value(self, y, w, Xw):
        """Compute the quantile Huber loss value."""
        n_samples = len(y)
        res = 0.0
        for i in range(n_samples):
            residual = y[i] - Xw[i]
            res += self._loss_sample(residual)
        return res / n_samples

    def _loss_sample(self, residual):
        """Calculate loss for a single sample."""
        tau = self.quantile
        delta = self.delta
        r = residual

        if r >= delta:
            # Upper linear tail: r >= delta
            return tau * (r - delta/2)
        elif r >= 0:
            # Upper quadratic: 0 <= r < delta
            return tau * r**2 / (2 * delta)
        elif r > -delta:
            # Lower quadratic: -delta < r < 0
            return (1 - tau) * r**2 / (2 * delta)
        else:
            # Lower linear tail: r <= -delta
            return (1 - tau) * (-r - delta/2)

    def gradient_scalar(self, X, y, w, Xw, j):
        """Compute gradient w.r.t. w_j - following parent class pattern."""
        n_samples = len(y)
        grad_j = 0.0
        for i in range(n_samples):
            residual = y[i] - Xw[i]
            grad_j += -X[i, j] * self._grad_per_sample(residual)
        return grad_j / n_samples

    def gradient_scalar_sparse(self, X_data, X_indptr, X_indices, y, Xw, j):
        """Compute gradient w.r.t. w_j for sparse matrices."""
        grad_j = 0.0
        for idx in range(X_indptr[j], X_indptr[j + 1]):
            i = X_indices[idx]
            residual = y[i] - Xw[i]
            grad_j += -X_data[idx] * self._grad_per_sample(residual)
        return grad_j / len(y)

    def _grad_per_sample(self, residual):
        """Calculate gradient for a single sample."""
        tau = self.quantile
        delta = self.delta
        r = residual

        if r >= delta:
            # Upper linear tail: r >= delta
            return tau
        elif r >= 0:
            # Upper quadratic: 0 <= r < delta
            return tau * r / delta
        elif r > -delta:
            # Lower quadratic: -delta < r < 0
            return (1 - tau) * r / delta
        else:
            # Lower linear tail: r <= -delta
            return tau - 1

    def full_grad_sparse(self, X_data, X_indptr, X_indices, y, Xw):
        """Compute full gradient for sparse matrices."""
        n_features = X_indptr.shape[0] - 1
        grad = np.zeros(n_features, dtype=X_data.dtype)

        for j in range(n_features):
            grad_j = 0.0
            for idx in range(X_indptr[j], X_indptr[j + 1]):
                i = X_indices[idx]
                residual = y[i] - Xw[i]
                grad_j += -X_data[idx] * self._grad_per_sample(residual)
            grad[j] = grad_j / len(y)

        return grad

    def get_lipschitz(self, X, y):
        n_features = X.shape[1]

        lipschitz = np.zeros(n_features, dtype=X.dtype)
        c = max(self.quantile, 1 - self.quantile) / self.delta
        for j in range(n_features):
            lipschitz[j] = c * (X[:, j] ** 2).sum() / len(y)

        return lipschitz

    def get_lipschitz_sparse(self, X_data, X_indptr, X_indices, y):
        """Compute Lipschitz constants for sparse matrices."""
        n_features = len(X_indptr) - 1
        lipschitz = np.zeros(n_features, dtype=X_data.dtype)
        c = max(self.quantile, 1 - self.quantile) / self.delta

        for j in range(n_features):
            nrm2 = 0.0
            for idx in range(X_indptr[j], X_indptr[j + 1]):
                nrm2 += X_data[idx] ** 2
            lipschitz[j] = c * nrm2 / len(y)

        return lipschitz

    def get_global_lipschitz(self, X, y):
        c = max(self.quantile, 1 - self.quantile) / self.delta
        return c * norm(X, ord=2) ** 2 / len(y)

    def get_global_lipschitz_sparse(self, X_data, X_indptr, X_indices, y):
        """Compute global Lipschitz constant for sparse matrices."""
        c = max(self.quantile, 1 - self.quantile) / self.delta
        spectral_norm_X = spectral_norm(X_data, X_indptr, X_indices, len(y))
        return c * spectral_norm_X ** 2 / len(y)

    def intercept_update_step(self, y, Xw):
        n_samples = len(y)

        # Compute gradient
        grad = 0.0
        for i in range(n_samples):
            residual = y[i] - Xw[i]
            grad -= self._grad_per_sample(residual)
        grad /= n_samples

        # Apply step size 1/c
        c = max(self.quantile, 1 - self.quantile) / self.delta
        return grad / c


class SmoothQuantileRegressor(BaseEstimator, RegressorMixin):
    """Quantile regression with progressive smoothing."""

    def __init__(self, quantile=0.5, alpha=0.1, delta_init=1.0, delta_final=1e-3,
                 n_deltas=10, max_iter=1000, tol=1e-6, verbose=False,
                 fit_intercept=True):
        self.quantile = quantile
        self.alpha = alpha
        self.delta_init = delta_init
        self.delta_final = delta_final
        self.n_deltas = n_deltas
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        """Fit using progressive smoothing: delta_init --> delta_final."""
        w = np.zeros(X.shape[1])
        deltas = np.geomspace(self.delta_init, self.delta_final, self.n_deltas)

        if self.verbose:
            print(
                f"Progressive smoothing: delta {self.delta_init:.2e} --> "
                f"{self.delta_final:.2e} in {self.n_deltas} steps")

        datafit = QuantileHuber(quantile=self.quantile, delta=self.delta_init)
        penalty = L1(alpha=self.alpha)

        # Use AndersonCD solver
        solver = AndersonCD(max_iter=self.max_iter, tol=self.tol,
                            warm_start=True, fit_intercept=self.fit_intercept,
                            verbose=max(0, self.verbose - 1))

        est = GeneralizedLinearEstimator(
            datafit=datafit, penalty=penalty, solver=solver)

        for i, delta in enumerate(deltas):
            datafit.delta = float(delta)

            est.fit(X, y)
            w = est.coef_.copy()

            if self.verbose:
                residuals = y - X @ w
                if self.fit_intercept:
                    residuals -= est.intercept_
                pinball_loss = np.mean(residuals * (self.quantile - (residuals < 0)))

                print(
                    f"  Stage {i+1:2d}: delta={delta:.2e}, "
                    f"pinball_loss={pinball_loss:.6f}, "
                    f"n_iter={est.n_iter_}"
                )

        self.est_ = est
        self.coef_ = est.coef_
        if self.fit_intercept:
            self.intercept_ = est.intercept_

        return self

    def predict(self, X):
        """Predict using the fitted model."""
        if not hasattr(self, "est_"):
            raise NotFittedError(
                "This SmoothQuantileRegressor instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )
        return self.est_.predict(X)
