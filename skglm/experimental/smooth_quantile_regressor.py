import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array
from ..solvers import FISTA
from ..penalties import L1
from ..estimators import GeneralizedLinearEstimator
from .quantile_huber import QuantileHuber


class SmoothQuantileRegressor(BaseEstimator, RegressorMixin):
    """Quantile regression with progressive smoothing using Huberized loss."""

    def __init__(self, quantile=0.75, alpha=1e-8, max_iter=1000, tol=1e-6,
                 delta_init=1.0, delta_final=1e-4, n_deltas=10, fit_intercept=True):
        self.quantile = quantile
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.delta_init = delta_init
        self.delta_final = delta_final
        self.n_deltas = n_deltas
        self.fit_intercept = fit_intercept
        self.intercept_ = 0.0

    def fit(self, X, y):
        """Fit using FISTA with decreasing smoothing parameter delta.

        For each delta level:
        - Update coefficients using FISTA
        - Update intercept using gradient step
        """
        X, y = check_X_y(X, y)
        w = np.zeros(X.shape[1])
        intercept = np.quantile(y, self.quantile) if self.fit_intercept else 0.0

        for delta in np.geomspace(self.delta_init, self.delta_final, self.n_deltas):
            datafit = QuantileHuber(quantile=self.quantile, delta=delta)
            est = GeneralizedLinearEstimator(
                datafit=datafit,
                penalty=L1(alpha=self.alpha),
                solver=FISTA(max_iter=self.max_iter, tol=self.tol)
            )
            est.coef_ = w
            est.fit(X, y)
            w = est.coef_

            if self.fit_intercept:
                pred = X @ w + intercept
                lipschitz = datafit.get_global_lipschitz(X, y)
                grad = np.mean(datafit.raw_grad(y, pred))
                intercept -= grad / lipschitz

            # Debug prints
            residuals = y - X.dot(w) - intercept
            obj_value = datafit.value(residuals, None, residuals) + \
                self.alpha * np.sum(np.abs(w))
            print(f"Delta: {delta:.6f}, Objective: {obj_value:.4f}, "
                  f"Intercept: {intercept:.4f}, "
                  f"Non-zero coefs: {np.sum(np.abs(w) > 1e-6)}, "
                  f"Lipschitz: {lipschitz:.4f}")
            print(f"Residual stats - mean: {np.mean(residuals):.4f}, "
                  f"std: {np.std(residuals):.4f}, "
                  f"min: {np.min(residuals):.4f}, "
                  f"max: {np.max(residuals):.4f}")

            coverage = np.mean(y <= X.dot(w) + intercept)
            print(f"Coverage: {coverage:.4f} (target: {self.quantile:.4f})")

        self.coef_, self.intercept_ = w, intercept
        return self

    def predict(self, X):
        """Predict using the fitted model."""
        check_array(X)
        return X @ self.coef_ + self.intercept_
