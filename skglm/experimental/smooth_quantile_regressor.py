import copy
import warnings
import numpy as np
from scipy import sparse

from skglm import GeneralizedLinearEstimator
from skglm.penalties import L1
from skglm.datafits import Huber
from skglm.experimental.quantile_regression import Pinball
from skglm.solvers import FISTA, LBFGS


class SmoothQuantileRegressor:
    """Progressive smoothing (homotopy) meta-solver.

    This solver addresses convergence issues in non-smooth datafits like Pinball
    (quantile regression) on large datasets (as discussed in GitHub issue #276).

    It works by progressively solving a sequence of smoothed problems with
    decreasing smoothing parameter.

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


    References
    ----------
    Chen, C. (2007). A Finite Smoothing Algorithm for Quantile Regression.
    Journal of Computational and Graphical Statistics, 16(1), 136–164.
    http://www.jstor.org/stable/27594233


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
        verbose=False,
    ):
        base_seq = smoothing_sequence or [1.0, 0.5, 0.2, 0.1, 0.05]

        # if user stops above min_delta, append finer deltas
        min_delta = 1e-3
        if base_seq[-1] > min_delta:
            extra = np.geomspace(base_seq[-1], min_delta, num=5, endpoint=False)[1:]
            base_seq = base_seq + list(extra)
        self.smoothing_sequence = base_seq
        self.quantile = float(quantile)

        if not 0 < self.quantile < 1:
            raise ValueError("quantile must be between 0 and 1")

        self.alpha = float(alpha)

        # default smooth solver: FISTA (proximal gradient for Huber + L1)
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

        if not hasattr(self.smooth_solver, 'warm_start'):
            self.smooth_solver.warm_start = False

        self.verbose = verbose

        # Always import QuantileHuber for all quantiles
        try:
            from skglm.experimental.quantile_huber import QuantileHuber
            self._quantile_huber_cls = QuantileHuber
        except ImportError:
            raise ImportError(
                "QuantileHuber class is required. "
                "Please ensure skglm.experimental.quantile_huber is available."
            )

    def fit(self, X, y):
        """Fit the model according to the given training data."""
        n_features = X.shape[1]
        coef = np.zeros(n_features)
        intercept = 0.0
        is_sparse = sparse.issparse(X)

        stage_results = []

        best_quantile_error = float('inf')
        best_coef = None
        best_intercept = None
        best_delta = None

        extended_sequence = self.smoothing_sequence[:]
        fine_deltas = [0.15, 0.12, 0.1, 0.08, 0.06, 0.04, 0.02, 0.01]
        extended_sequence = sorted(set(extended_sequence + fine_deltas), reverse=True)

        # Progressive smoothing stages
        for stage, delta in enumerate(extended_sequence):
            if self.verbose:
                print(f"[ProgressiveSmoothing] Stage {stage+1}/{len(extended_sequence)}: "
                      f"delta = {delta:.3g}")

            # Always use QuantileHuber for all quantile values
            datafit = self._quantile_huber_cls(delta=delta, quantile=self.quantile)

            solver = copy.deepcopy(self.smooth_solver)

            if not hasattr(solver, 'fit_intercept'):
                solver.fit_intercept = True

            est = GeneralizedLinearEstimator(
                datafit=datafit,
                penalty=L1(alpha=self.alpha),
                solver=solver,
            )

            if stage > 0:
                est.coef_ = coef
                est.intercept_ = intercept

            est.fit(X, y)

            coef, intercept = est.coef_, est.intercept_

            # Check quantile error
            y_pred = X @ coef if not is_sparse else X.dot(coef)
            residuals = y - y_pred
            actual_quantile = np.sum(residuals > 0) / len(residuals)
            quantile_error = abs(actual_quantile - self.quantile)

            if self.verbose:
                print(
                    f"  Actual quantile: {actual_quantile:.3f}, Error: {quantile_error:.3f}")

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
                'quantile_error': quantile_error,
                'actual_quantile': actual_quantile,
                'coef': coef.copy(),
                'intercept': intercept
            }

            # Add solver-specific info if available
            if hasattr(solver, "n_iter_"):
                stage_info['n_iter'] = solver.n_iter_

            stage_results.append(stage_info)

        # Always use the best smoothed solution - never use PDCD_WS
        self.coef_ = best_coef
        self.intercept_ = best_intercept

        if self.verbose:
            print(f"[Final] Using smoothed solution with delta={best_delta:.3g}")
            print(f"  Best quantile error: {best_quantile_error:.3f}")

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
