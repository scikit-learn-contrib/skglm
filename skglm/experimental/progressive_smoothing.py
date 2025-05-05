"""
Progressive smoothing solver for non-smooth datafits (e.g., Pinball loss).

This is a meta-solver that gradually approximates a non-smooth loss with a
smooth version (e.g., Huber-smoothed Pinball), solves the problem at each
smoothing level with a smooth solver, and eventually switches
to a non-smooth solver (PDCD_WS) once the smoothing parameter is sufficiently small.

References
----------
.. [1] Nesterov, Y.
       "Smooth minimization of non-smooth functions," Mathematical Programming, 2005.
       https://link.springer.com/article/10.1007/s10107-004-0552-5

.. [2] Beck, A. and Teboulle, M.
       "Smoothing and first order methods: A unified framework,"
       SIAM Journal on Optimization, 2012.
       https://epubs.siam.org/doi/10.1137/100818327
"""

import copy
import warnings
import numpy as np
from scipy import sparse

from skglm import GeneralizedLinearEstimator
from skglm.penalties import L1
from skglm.datafits import Huber
from skglm.experimental.pdcd_ws import PDCD_WS
from skglm.experimental.quantile_regression import Pinball
from skglm.solvers import LBFGS, FISTA

# ---- ensure Huber and L1 get a gradient method for LBFGS ----


def _huber_gradient(self, X, y, Xw):
    n_samples = len(y)
    r = y - Xw
    δ = self.delta
    # derivative of Huber: r/δ in |r|≤δ, sign(r) outside
    dr = np.where(np.abs(r) <= δ, r / δ, np.sign(r))
    return - X.T.dot(dr) / n_samples


def _l1_gradient(self, w):
    # simple subgradient of ‖w‖₁
    return self.alpha * np.sign(w)


if not hasattr(Huber, 'gradient'):
    Huber.gradient = _huber_gradient

if not hasattr(L1, 'gradient'):
    L1.gradient = _l1_gradient


class ProgressiveSmoothingSolver:
    """Progressive smoothing (homotopy) meta-solver.

    This solver addresses convergence issues in the PDCD_WS solver with
    non-smooth datafits like Pinball (quantile regression) on large datasets
    (as discussed in GitHub issue #276).

    It works by progressively solving a sequence of smoothed problems with
    decreasing smoothing parameter, and finally solving the original non-smooth
    problem using PDCD_WS initialized with the smoothed solution.

    Parameters
    ----------
    smoothing_sequence : list or None, default=None
        Sequence of decreasing smoothing parameters (Huber delta values).
        If None, defaults to [1.0, 0.5, 0.2, 0.1, 0.05].

    quantile : float, default=0.5
        Desired quantile level between 0 and 1. When 0.5, uses symmetric Huber.
        Otherwise, uses QuantileHuber for asymmetric smoothing.

    alpha : float, default=0.1
        L1 regularization strength.

    smooth_solver : instance of BaseSolver, default=None
        Solver to use for smooth approximation stages.
        If None, uses LBFGS(max_iter=500, tol=1e-6).

    nonsmooth_solver : instance of BaseSolver, default=None
        Solver to use for final non-smooth problem.
        If None, uses PDCD_WS with appropriate settings.

    verbose : bool, default=False
        If True, prints progress information during fitting.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Parameter vector (w in the cost function formula).

    intercept_ : float
        Intercept term in decision function.

    stage_results_ : list
        Information about each smoothing stage including smoothing parameter,
        number of iterations, and coefficients.

    Examples
    --------
    >>> from skglm.experimental.progressive_smoothing import ProgressiveSmoothingSolver
    >>> import numpy as np
    >>> X = np.random.randn(1000, 10)
    >>> y = np.random.randn(1000)
    >>> solver = ProgressiveSmoothingSolver(quantile=0.8)
    >>> solver.fit(X, y)
    >>> print(solver.coef_)
    """

    def __init__(
        self,
        smoothing_sequence=None,
        quantile=0.5,
        alpha=0.1,
        smooth_solver=None,
        nonsmooth_solver=None,
        verbose=False,
    ):
        # Build and *extend* the smoothing sequence
        base_seq = smoothing_sequence or [1.0, 0.5, 0.2, 0.1, 0.05]

        # For asymmetric quantiles, extend the sequence more aggressively
        if abs(quantile - 0.5) > 0.1:  # If not close to median
            min_delta = 1e-4  # Go much further down for asymmetric quantiles
        else:
            min_delta = 1e-3  # Original minimum for symmetric case

        # if user stops above min_delta, append finer deltas
        if base_seq[-1] > min_delta:
            extra = np.geomspace(base_seq[-1], min_delta, num=5, endpoint=False)[1:]
            base_seq = base_seq + list(extra)
        self.smoothing_sequence = base_seq
        self.quantile = float(quantile)

        if not 0 < self.quantile < 1:
            raise ValueError("quantile must be between 0 and 1")

        self.alpha = float(alpha)
        # self.smooth_solver = smooth_solver or LBFGS(max_iter=500, tol=1e-6)
        # # LBFGS in skglm does not properly handle an intercept column ⇒ disable it
        # if isinstance(self.smooth_solver, LBFGS):
        #     self.smooth_solver.fit_intercept = False
        # default smooth solver: FISTA (proximal gradient for Huber + ℓ₁)
        if smooth_solver is None:
            self.smooth_solver = FISTA(max_iter=2000, tol=1e-8)
        else:
            # if they passed LBFGS, override it
            if isinstance(smooth_solver, LBFGS):
                import warnings
                warnings.warn(
                    "Overriding provided LBFGS: using FISTA "
                    "for L1‐regularized smoothing stages."
                )
                self.smooth_solver = FISTA(
                    max_iter=max(2000, smooth_solver.max_iter),
                    tol=min(1e-8, smooth_solver.tol),
                )
            else:
                self.smooth_solver = smooth_solver
        self.smooth_solver.fit_intercept = False
        # ensure a warm_start flag on every solver
        if not hasattr(self.smooth_solver, 'warm_start'):
            self.smooth_solver.warm_start = False
        self.nonsmooth_solver = nonsmooth_solver or PDCD_WS(
            max_iter=10000,
            max_epochs=5000,
            # p0=500,
            tol=1e-8,
            warm_start=True,
            verbose=verbose,
        )
        self.nonsmooth_solver.fit_intercept = True
        if not hasattr(self.nonsmooth_solver, 'warm_start'):
            self.nonsmooth_solver.warm_start = False
        self.verbose = verbose

        # Check if we need QuantileHuber for asymmetric quantiles
        if self.quantile != 0.5:
            try:
                from skglm.experimental.quantile_huber import QuantileHuber
                self._quantile_huber_cls = QuantileHuber
            except ImportError:
                raise ImportError(
                    "QuantileHuber class is required for quantile != 0.5. "
                    "Please ensure skglm.experimental.quantile_huber is available."
                )
        else:
            self._quantile_huber_cls = None

    # Optimal delta approach: Find the smoothing stage that gives the best quantile

    def fit(self, X, y):
        """Fit the model according to the given training data."""
        # Initialize coefficients
        n_features = X.shape[1]
        coef = np.zeros(n_features)
        intercept = 0.0
        is_sparse = sparse.issparse(X)

        # Track progress of each stage
        stage_results = []

        # For asymmetric quantiles, track the best solution
        best_quantile_error = float('inf')
        best_coef = None
        best_intercept = None
        best_delta = None

        # Extended smoothing sequence
        extended_sequence = self.smoothing_sequence[:]
        # Add finer steps specifically to capture the optimal delta
        if abs(self.quantile - 0.5) > 0.05:
            # Add a range around 0.1 where we saw good quantile accuracy
            fine_deltas = [0.15, 0.12, 0.1, 0.08, 0.06, 0.04, 0.02, 0.01]
            extended_sequence = sorted(
                set(extended_sequence + fine_deltas), reverse=True)

        # Progressive smoothing stages
        for stage, delta in enumerate(extended_sequence):
            if self.verbose:
                print(f"[ProgressiveSmoothing] Stage {stage+1}/{len(extended_sequence)}: "
                      f"delta = {delta:.3g}")

            # Create appropriate datafit based on quantile
            if self.quantile == 0.5:
                datafit = Huber(delta=delta)
            else:
                datafit = self._quantile_huber_cls(delta=delta, quantile=self.quantile)

            # Create a modified copy of the solver
            solver = copy.deepcopy(self.smooth_solver)

            # Safely set fit_intercept attribute if the solver can accept it
            if not hasattr(solver, 'fit_intercept'):
                solver.fit_intercept = True

            # Create estimator
            est = GeneralizedLinearEstimator(
                datafit=datafit,
                penalty=L1(alpha=self.alpha),
                solver=solver,
            )

            # Warm start if available and possible
            if stage > 0:
                est.coef_ = coef
                est.intercept_ = intercept

            # Fit model
            est.fit(X, y)

            # Store results
            coef, intercept = est.coef_, est.intercept_

            # Check quantile error for asymmetric quantiles
            if abs(self.quantile - 0.5) > 0.05:
                y_pred = X @ coef if not is_sparse else X.dot(coef)
                residuals = y - y_pred
                actual_quantile = np.sum(residuals > 0) / len(residuals)
                quantile_error = abs(actual_quantile - self.quantile)

                if self.verbose:
                    print(
                        f"  Actual quantile: {actual_quantile:.3f}, Error: {quantile_error:.3f}")

                # Update best solution if this is better
                if quantile_error < best_quantile_error:
                    best_quantile_error = quantile_error
                    best_coef = coef.copy()
                    best_intercept = intercept
                    best_delta = delta

                    if self.verbose:
                        print(
                            f"  New best quantile at delta={delta:.3g}, error={quantile_error:.3f}")

            # Record stage information
            obj_value = datafit.value(
                y, coef, X @ coef if not is_sparse else X.dot(coef))
            obj_value += est.penalty.value(coef)

            stage_info = {
                'delta': delta,
                'obj_value': obj_value,
                'coef_norm': np.linalg.norm(coef),
            }

            # Add solver-specific info if available
            if hasattr(solver, "n_iter_"):
                stage_info['n_iter'] = solver.n_iter_

            stage_results.append(stage_info)

        # Handle final stage based on quantile type
        if abs(self.quantile - 0.5) > 0.05:  # Asymmetric quantile
            # Use the best smoothed solution (skip PDCD_WS entirely)
            self.coef_ = best_coef
            self.intercept_ = best_intercept

            if self.verbose:
                print(f"[Final] Using smoothed solution with delta={best_delta:.3g}")
                print(f"  Best quantile error: {best_quantile_error:.3f}")
        else:
            # For symmetric quantiles, still use PDCD_WS
            if self.verbose:
                print(
                    "[ProgressiveSmoothing] Final stage: non-smooth solver on true Pinball loss")

            final_solver = copy.deepcopy(self.nonsmooth_solver)
            # Ensure it has fit_intercept attribute
            if not hasattr(final_solver, 'fit_intercept'):
                final_solver.fit_intercept = True

            final_est = GeneralizedLinearEstimator(
                datafit=Pinball(self.quantile),
                penalty=L1(alpha=self.alpha),
                solver=final_solver,
            )

            # Initialize with smoothed solution if possible
            final_est.coef_ = coef
            final_est.intercept_ = intercept

            # Fit with non-smooth solver
            final_est.fit(X, y)

            # Store final results
            self.coef_ = final_est.coef_
            self.intercept_ = final_est.intercept_

        self.stage_results_ = stage_results

        return self

    def predict(self, X):
        """Predict using the linear model.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array, shape (n_samples,)
            Returns predicted values.
        """
        if not hasattr(self, "coef_"):
            raise ValueError("Model not fitted. Call fit before predict.")

        is_sparse = sparse.issparse(X)
        if is_sparse:
            return X.dot(self.coef_) + self.intercept_
        else:
            return X @ self.coef_ + self.intercept_
