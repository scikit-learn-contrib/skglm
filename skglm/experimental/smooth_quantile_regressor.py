import numpy as np
from scipy import sparse
from skglm.experimental.pdcd_ws import PDCD_WS
from numba import jit
from skglm import GeneralizedLinearEstimator
from skglm.penalties import L1
from skglm.experimental.solver_strategies import StageBasedSolverStrategy


@jit(nopython=True)
def compute_quantile_error(residuals, target_quantile):
    """Compute quantile error with stronger quantile enforcement."""
    n_samples = len(residuals)
    actual_quantile = np.sum(residuals < 0) / n_samples
    # Add penalty for deviation from target quantile
    quantile_error = abs(actual_quantile - target_quantile)
    # Add additional penalty for wrong direction
    if (target_quantile < 0.5 and actual_quantile > target_quantile) or \
       (target_quantile > 0.5 and actual_quantile < target_quantile):
        quantile_error *= 2.0  # Double the error if in wrong direction
    return quantile_error


@jit(nopython=True)
def compute_adaptive_delta(current_delta, quantile_error, gap, min_delta=1e-8):
    """Compute next delta value with stronger quantile enforcement."""
    if quantile_error < 0.005 and gap < 0.05:
        # Very close to solution, reduce slowly
        return max(current_delta * 0.8, min_delta)
    elif quantile_error < 0.02 and gap < 0.1:
        # Getting closer, reduce moderately
        return max(current_delta * 0.85, min_delta)
    else:
        # Far from solution, reduce very conservatively
        return max(current_delta * 0.9, min_delta)


@jit(nopython=True, cache=True)
def max_subgrad_gap(residuals, delta, quantile):
    """Compute the maximum subgradient gap for stopping criteria using numba."""
    small_residuals_mask = np.abs(residuals) <= delta

    # Skip computation if no residuals within delta
    if not np.any(small_residuals_mask):
        return 0.0

    small_residuals = residuals[small_residuals_mask]

    # Compute gaps for all small residuals at once
    gaps = np.zeros_like(small_residuals)
    pos_mask = small_residuals >= 0
    neg_mask = ~pos_mask

    # Handle positive residuals
    if np.any(pos_mask):
        pos_r = small_residuals[pos_mask]
        gaps[pos_mask] = np.abs(quantile * pos_r/delta - quantile)

    # Handle negative residuals
    if np.any(neg_mask):
        neg_r = small_residuals[neg_mask]
        term1 = (1-quantile) * neg_r/delta
        term2 = (quantile-1)
        gaps[neg_mask] = np.abs(term1 - term2)

    return gaps.max() if len(gaps) > 0 else 0.0


class SmoothQuantileRegressor:
    r"""Progressive smoothing solver for quantile regression.

    This solver addresses convergence issues in non-smooth quantile regression
    on large datasets by using a progressive smoothing approach. The optimization
    objective is:

    .. math::
        \min_{w \in \mathbb{R}^p} \frac{1}{n} \sum_{i=1}^n \rho_\tau(y_i - x_i^T w)
        + \alpha \|w\|_1

    where :math:`\rho_\tau` is the pinball loss:

    .. math::

       \rho_\tau(r) =
       \begin{cases}
           \tau\, r, & \text{if } r \ge 0,\\
           (\tau - 1)\, r, & \text{if } r < 0.
       \end{cases}

    The solver progressively approximates the non-smooth pinball loss using
    smoothed versions with decreasing smoothing parameter :math:`\delta`:

    .. math::

       \rho_\tau^\delta(r) =
       \begin{cases}
           \tau\, r - \dfrac{\delta}{2}, & \text{if } r \ge \delta,\\
           \dfrac{r^2}{2\delta}, & \text{if } |r| < \delta,\\
           (\tau - 1)\, r - \dfrac{\delta}{2}, & \text{if } r \le -\delta.
       \end{cases}

    Parameters
    ----------
    smoothing_sequence : list, default=None
        List of smoothing parameters (Huber delta values).
        If None, uses adaptive sequence.

    quantile : float, default=0.5
        Desired quantile level between 0 and 1. When 0.5, uses symmetric Huber.
        Otherwise, uses QuantileHuber for asymmetric smoothing.

    alpha : float, default=0.1
        L1 regularization strength.

    initial_delta : float, default=0.5
        Initial smoothing parameter (Huber delta value).

    min_delta : float, default=1e-6
        Minimum smoothing parameter (Huber delta value).

    smooth_solver : instance of BaseSolver, default=None
        Solver to use for smooth approximation stages.
        If None, uses PDCD_WS with optimized parameters.

    verbose : bool, default=False
        If True, prints progress information during fitting.

    delta_tol : float, default=1e-6
        Tolerance for the maximum subgradient gap.

    max_stages : int, default=8
        Maximum number of smoothing stages.

    quantile_error_threshold : float, default=0.005
        Threshold for quantile error to stop fitting.

    solver_params : dict, default=None
        Dictionary of parameters for configuring solver behavior. Available parameters:

        - 'base_tol': float, default=1e-4
          Base tolerance for solvers.
        - 'tol_delta_factor': float, default=0.1
          Factor to multiply delta by when computing tolerance.
        - 'max_iter_start': int, default=100
          Maximum iterations for the first stage.
        - 'max_iter_step': int, default=50
          Additional iterations to add for each subsequent stage.
        - 'max_iter_cap': int, default=1000
          Maximum iterations cap for any stage.
        - 'large_problem_threshold': int, default=1000
          Threshold for considering a problem "large" in terms of features.
        - 'small_problem_threshold': int, default=100
          Threshold for considering a problem "small" in terms of features.
        - 'p0_frac_large': float, default=0.1
          Fraction of features to use in working set for large problems.
        - 'p0_frac_small': float, default=0.5
          Fraction of features to use in working set for small problems.
        - 'p0_min': int, default=10
          Minimum working set size for any problem.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Parameter vector (:math:`w` in the cost function formula).

    intercept_ : float
        Intercept term in decision function.

    stage_results_ : list
        Information about each smoothing stage including:
        - delta: smoothing parameter
        - obj_value: objective value
        - coef_norm: L2 norm of coefficients
        - quantile_error: absolute difference between target and achieved quantile
        - actual_quantile: achieved quantile level
        - coef: coefficient vector
        - intercept: intercept value
        - n_iter: number of iterations (if available)

    Notes
    -----
    This implementation uses a progressive smoothing approach to solve quantile
    regression problems. It starts with a highly smoothed approximation and
    gradually reduces the smoothing parameter to approach the original non-smooth
    problem. This approach is particularly effective for large datasets where
    direct optimization of the non-smooth objective can be challenging.

    The solver automatically selects the best solution from all smoothing stages
    based on the quantile error, ensuring good approximation of the target
    quantile level.

    References
    ----------
    Chen, C. (2007). A Finite Smoothing Algorithm for Quantile Regression.
    Journal of Computational and Graphical Statistics, 16(1), 136–164.
    http://www.jstor.org/stable/27594233

    Examples
    --------
    >>> from skglm.experimental.smooth_quantile_regressor import SmoothQuantileRegressor
    >>> import numpy as np
    >>> X = np.random.randn(1000, 10)
    >>> y = np.random.randn(1000)
    >>> reg = SmoothQuantileRegressor(quantile=0.8, alpha=0.1)
    >>> reg.fit(X, y)
    >>> print(reg.coef_)
    """

    def __init__(
        self,
        smoothing_sequence=None,
        quantile=0.5,
        alpha=0.1,
        initial_delta=0.5,
        min_delta=1e-6,
        smooth_solver=None,
        verbose=False,
        delta_tol=1e-6,
        max_stages=8,
        quantile_error_threshold=0.005,
        solver_params=None,
        initial_alpha: float = None,
        alpha_schedule: str = 'geometric',
    ):
        self.quantile = float(quantile)
        if not 0 < self.quantile < 1:
            raise ValueError("quantile must be between 0 and 1")
        self.alpha = float(alpha)
        self.initial_delta = float(initial_delta)
        self.min_delta = float(min_delta)
        self.verbose = verbose
        self.delta_tol = float(delta_tol)
        self.max_stages = int(max_stages)
        self.quantile_error_threshold = float(quantile_error_threshold)
        self.smoothing_sequence = smoothing_sequence

        # L1‐penalty continuation: start at initial_alpha, end at self.alpha
        self.initial_alpha = float(
            initial_alpha) if initial_alpha is not None else 5.0 * self.alpha
        self.alpha_schedule = alpha_schedule

        self.solver_params = {} if solver_params is None else dict(solver_params)

        # Initialize solver strategy
        self.solver_strategy = StageBasedSolverStrategy(self.solver_params)

        from skglm.experimental.quantile_huber import QuantileHuber
        self._quantile_huber_cls = QuantileHuber

        # Initialize solver - use PDCD for L1 regularization
        if smooth_solver is None:
            self.smooth_solver = PDCD_WS(
                max_iter=200,
                tol=1e-8,
                fit_intercept=False,
                warm_start=True,
                p0=200
            )
        else:
            self.smooth_solver = smooth_solver

        # Ensure consistent solver settings
        if not hasattr(self.smooth_solver, 'warm_start'):
            self.smooth_solver.warm_start = False

    def _initialize_pdcd(self, X, y):
        """Initialize PDCD solver with good primal and dual variables."""
        _, n_features = X.shape

        if hasattr(self, 'coef_') and self.coef_ is not None:
            w = self.coef_.copy()
        else:
            w = np.zeros(n_features)

        residuals = y - X.dot(w)
        dual = np.where(
            residuals > 0,
            self.quantile,
            self.quantile - 1
        )

        return w, dual

    def _get_solver_for_stage(self, delta, stage, n_features):
        """Get solver with parameters adapted to current stage.

        Uses the solver strategy to create and configure a solver appropriate
        for the current smoothing stage.

        Parameters
        ----------
        delta : float
            Current smoothing parameter.
        stage : int
            Current stage number (0-indexed).
        n_features : int
            Number of features in the dataset.

        Returns
        -------
        solver : BaseSolver
            Configured solver instance for the current stage.
        """
        # Get base solver configuration
        solver = self.solver_strategy.create_solver_for_stage(
            self.smooth_solver, delta, stage, n_features)

        # Additional quantile-specific configuration
        if hasattr(solver, 'tol'):
            # Scale tolerance based on quantile error
            solver.tol = min(solver.tol, self.quantile_error_threshold * 0.1)

        return solver

    def fit(self, X, y):
        """Fit the model according to the given training data."""
        # Basic validation
        n_samples, n_features = X.shape
        if len(y) != n_samples:
            raise ValueError(f"X has {n_samples} samples, but y has {len(y)} samples")
        is_sparse = sparse.issparse(X)

        # Center data to handle intercept manually
        y_mean = np.mean(y)
        if is_sparse:
            X = X.toarray()
        X_mean = np.mean(X, axis=0)
        X = X - X_mean
        y = y - y_mean

        stage_results = []
        best_obj_value = float('inf')
        best_coef = None
        best_delta = None

        # Initialize with a solution that satisfies the quantile constraint
        if not hasattr(self, 'coef_') or self.coef_ is None:
            # Start with a solution that has roughly the right quantile
            sorted_y = np.sort(y)
            target_idx = int(self.quantile * n_samples)
            target_value = sorted_y[target_idx]
            self.coef_ = np.zeros(n_features)
            self.intercept_ = target_value

        # Choose between fixed or adaptive smoothing sequence
        is_fixed_sequence = self.smoothing_sequence is not None
        if is_fixed_sequence:
            deltas = list(self.smoothing_sequence)
        else:
            deltas = [self.initial_delta]

        # Build L1‐continuation schedule
        if self.alpha_schedule == 'geometric':
            alpha_seq = [
                self.initial_alpha *
                (self.alpha / self.initial_alpha) ** (i / max(self.max_stages - 1, 1))
                for i in range(self.max_stages)
            ]
        else:
            alpha_seq = [
                self.initial_alpha + (self.alpha - self.initial_alpha) *
                (i / max(self.max_stages - 1, 1))
                for i in range(self.max_stages)
            ]

        # Initialize solver variables
        coef, dual = self._initialize_pdcd(X, y)
        prev_coef = None
        last_gap = 1.0
        last_quantile_error = 1.0

        # Progressive smoothing stages
        for stage in range(self.max_stages):
            try:
                # Determine current delta
                if is_fixed_sequence:
                    if stage >= len(deltas):
                        break  # Exhausted fixed sequence
                    current_delta = deltas[stage]
                else:
                    if stage == 0:
                        current_delta = self.initial_delta
                    else:
                        # Compute next delta adaptively
                        current_delta = compute_adaptive_delta(
                            deltas[-1], last_quantile_error, last_gap,
                            min_delta=self.min_delta
                        )

                        # Stop if we've reached minimum delta
                        if current_delta <= self.min_delta:
                            if self.verbose:
                                print(f"  Reached minimum delta: {current_delta:.3g}")
                            break

                        # Stop if delta isn't changing significantly
                        delta_change = abs(current_delta - deltas[-1])
                        if delta_change < (self.min_delta * 0.1):
                            if self.verbose:
                                print(f"  Delta not changing significantly: "
                                      f"{current_delta:.3g}")
                            break

                        # Add to adaptive sequence
                        deltas.append(current_delta)

                if self.verbose:
                    print(f"[Stage {stage+1}/{self.max_stages}] "
                          f"delta={current_delta:.3g}")

                # Skip if coefficients haven't changed significantly
                if prev_coef is not None and stage > 0:
                    if np.allclose(coef, prev_coef, rtol=1e-5, atol=1e-7):
                        if self.verbose:
                            print("  Coefficients haven't changed, skipping stage")
                        continue

                # Get solver with adapted parameters
                solver = self._get_solver_for_stage(current_delta, stage, n_features)

                # Select penalty strength for this stage
                stage_alpha = alpha_seq[min(stage, len(alpha_seq) - 1)]

                # Setup datafit with quantile constraint
                datafit = self._quantile_huber_cls(
                    delta=current_delta, quantile=self.quantile)
                est = GeneralizedLinearEstimator(
                    datafit=datafit,
                    penalty=L1(alpha=stage_alpha),
                    solver=solver,
                )
                est.intercept_ = 0.0

                # Warm start primal/dual whenever available
                if stage > 0:
                    est.coef_ = coef
                if hasattr(solver, 'dual_init'):
                    solver.dual_init = dual

                # Fit with quantile constraint
                est.fit(X, y)

                # Extract results
                coef = est.coef_
                y_pred = X @ coef
                residuals = y - y_pred

                # Update dual variables with quantile constraint
                dual = np.where(residuals > 0, self.quantile, self.quantile - 1)

                # Calculate quantile metrics
                actual_quantile = np.sum(residuals < 0) / n_samples
                quantile_error = compute_quantile_error(residuals, self.quantile)
                last_quantile_error = quantile_error

                # Compute true pinball loss
                pin_loss = np.mean(
                    np.where(residuals >= 0,
                             self.quantile * residuals,
                             (self.quantile - 1) * residuals)
                )
                obj_value_true = pin_loss + est.penalty.value(coef)

                # Record stage information
                stage_info = {
                    'delta': current_delta,
                    'obj_value': obj_value_true,
                    'coef_norm': np.linalg.norm(coef),
                    'quantile_error': quantile_error,
                    'actual_quantile': actual_quantile,
                    'true_loss': pin_loss,
                    'intercept': float(y_mean - np.dot(X_mean, coef))
                }

                # Add iteration count if available
                if hasattr(solver, "n_iter_"):
                    stage_info['n_iter'] = solver.n_iter_

                stage_results.append(stage_info)

                # Update best solution based on true penalized objective (fit + L1)
                if obj_value_true < best_obj_value:
                    best_obj_value = obj_value_true
                    best_coef = coef.copy()
                    best_delta = current_delta

                    if self.verbose:
                        print(f"  New best solution (stage {stage+1})")

                # Compute stopping criteria
                gap = max_subgrad_gap(residuals, current_delta, self.quantile)
                last_gap = gap

                if self.verbose:
                    print(f"  Max subgradient gap: {gap:.3g}")

                # Early stopping with stronger quantile enforcement
                early_stop = (
                    gap <= self.delta_tol and
                    quantile_error < self.quantile_error_threshold and
                    # Stronger quantile constraint
                    abs(actual_quantile - self.quantile) <
                    self.quantile_error_threshold and
                    stage > 0
                )
                if early_stop:
                    if self.verbose:
                        print(f"  Early stopping: gap={gap:.3g} <= {self.delta_tol}")
                        print(f"  Quantile error: {quantile_error:.3f} < "
                              f"{self.quantile_error_threshold:.3f}")
                    break

            except Exception as e:
                if self.verbose:
                    print(f"  Error in stage {stage+1}: {str(e)}")
                    print("  Continuing with best solution found so far")
                break

        # Enforce sparsity by zeroing small coefficients
        threshold = self.alpha  # adjust this tolerance to control sparsity
        small_mask = np.abs(best_coef) < threshold
        if small_mask.any():
            best_coef = best_coef.copy()
            best_coef[small_mask] = 0.0
            if self.verbose:
                print(f"  Zeroed {small_mask.sum()} coefficients below {threshold}")

        # Calibrate intercept so that P(y <= Xw + intercept) = quantile
        y_orig = y + y_mean  # recover original y
        X_orig = X + X_mean  # recover original X
        res = y_orig - X_orig.dot(best_coef)
        # Use the tau-th percentile of residuals so P(res <= intercept) = tau
        self.intercept_ = float(np.percentile(res, self.quantile * 100))
        self.coef_ = best_coef

        if self.verbose:
            print(f"[Final] Using solution with delta={best_delta:.3g}")
            if not is_fixed_sequence:
                print(f"  Final delta sequence: {deltas}")

        self.stage_results_ = stage_results
        return self

    def predict(self, X):
        """Predict using the linear model."""
        if not hasattr(self, "coef_"):
            raise ValueError("Model not fitted. Call fit before predict.")

        is_sparse = sparse.issparse(X)
        if is_sparse:
            return X.dot(self.coef_) + self.intercept_
        else:
            return X @ self.coef_ + self.intercept_
