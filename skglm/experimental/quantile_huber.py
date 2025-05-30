import numpy as np
from numpy.linalg import norm
from numba import float64
from skglm.datafits.base import BaseDatafit
from sklearn.base import BaseEstimator, RegressorMixin
from skglm.solvers import FISTA, AndersonCD
from skglm.penalties import L1
from skglm.estimators import GeneralizedLinearEstimator


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
        Width of quadratic region.
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

    def get_lipschitz(self, X, y):
        n_features = X.shape[1]

        lipschitz = np.zeros(n_features, dtype=X.dtype)
        c = max(self.quantile, 1 - self.quantile) / self.delta
        for j in range(n_features):
            lipschitz[j] = c * (X[:, j] ** 2).sum() / len(y)

        return lipschitz

    def get_global_lipschitz(self, X, y):
        c = max(self.quantile, 1 - self.quantile) / self.delta
        return c * norm(X, ord=2) ** 2 / len(y)


class SmoothQuantileRegressor(BaseEstimator, RegressorMixin):
    """Quantile regression with progressive smoothing."""

    def __init__(self, quantile=0.5, alpha=0.1, delta_init=1.0, delta_final=1e-3,
                 n_deltas=10, max_iter=1000, tol=1e-4, verbose=False, solver="FISTA"):
        self.quantile = quantile
        self.alpha = alpha
        self.delta_init = delta_init
        self.delta_final = delta_final
        self.n_deltas = n_deltas
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.solver = solver

    def fit(self, X, y):
        """Fit using progressive smoothing: delta_init --> delta_final."""
        w = np.zeros(X.shape[1])
        deltas = np.geomspace(self.delta_init, self.delta_final, self.n_deltas)

        if self.verbose:
            print(
                f"Progressive smoothing: delta {self.delta_init:.3f} --> "
                f"{self.delta_final:.3f} in {self.n_deltas} steps")

        datafit = QuantileHuber(quantile=self.quantile, delta=self.delta_init)
        penalty = L1(alpha=self.alpha)
        # Solver selection
        if isinstance(self.solver, str):
            if self.solver == "FISTA":
                solver = FISTA(max_iter=self.max_iter, tol=self.tol)
                solver.warm_start = True
            elif self.solver == "AndersonCD":
                solver = AndersonCD(max_iter=self.max_iter, tol=self.tol,
                                    warm_start=True, fit_intercept=False)
            else:
                raise ValueError(f"Unknown solver: {self.solver}")
        else:
            solver = self.solver

        est = GeneralizedLinearEstimator(
            datafit=datafit, penalty=penalty, solver=solver)

        for i, delta in enumerate(deltas):
            datafit.delta = float(delta)

            est.fit(X, y)
            w = est.coef_.copy()

            if self.verbose:
                residuals = y - X @ w
                coverage = np.mean(residuals <= 0)
                pinball_loss = np.mean(residuals * (self.quantile - (residuals < 0)))

                print(
                    f"  Stage {i+1:2d}: delta={delta:.4f}, "
                    f"coverage={coverage:.3f}, pinball_loss={pinball_loss:.6f}, "
                    f"n_iter={est.n_iter_}"
                )

        self.est = est

        return self

    def predict(self, X):
        """Predict using the fitted model."""
        if not hasattr(self, "est"):
            raise ValueError("Call 'fit' before 'predict'.")
        return self.est.predict(X)
