import numpy as np
from numba import float64
from skglm.datafits import BaseDatafit
from skglm.penalties import L1
from skglm.solvers import AndersonCD, ProxNewton
from skglm.experimental import PDCD_WS
import time


class QuantileHuber(BaseDatafit):
    """Smoothed pinball loss using Huber approximation.

    The datafit transitions smoothly from quadratic to linear behavior
    at a threshold controlled by the parameter rho.
    """

    def __init__(self, quantile_level, rho=1.0):
        self.quantile_level = quantile_level
        self.rho = rho

    def value(self, y, w, Xw):
        residual = y - Xw
        q = self.quantile_level

        # For each residual, apply appropriate function based on magnitude
        loss = np.zeros_like(residual)

        # Positive residuals (y > Xw)
        pos_idx = residual > self.rho
        pos_smooth_idx = (0 < residual) & (residual <= self.rho)

        # Negative residuals (y < Xw)
        neg_idx = residual < -self.rho
        neg_smooth_idx = (-self.rho <= residual) & (residual <= 0)

        # Linear regions
        loss[pos_idx] = q * residual[pos_idx]
        loss[neg_idx] = (q - 1) * residual[neg_idx]

        # Quadratic transition regions
        loss[pos_smooth_idx] = q * (residual[pos_smooth_idx]
                                    ** 2 / (2 * self.rho) + self.rho/2)
        loss[neg_smooth_idx] = (q - 1) * (residual[neg_smooth_idx]
                                          ** 2 / (2 * self.rho) - self.rho/2)

        return np.sum(loss)

    def prox(self, w, step, y):
        """Prox of step * quantile_huber."""
        shift_cst = (self.quantile_level - 1/2) * step

        # Huber-like proximal mapping that depends on rho
        result = np.zeros_like(w)
        residual = y - w - shift_cst

        # Different regions based on residual magnitude
        pos_idx = residual > self.rho * step
        neg_idx = residual < -self.rho * step
        smooth_idx = ~(pos_idx | neg_idx)

        # Linear regions - use soft thresholding
        result[pos_idx] = y[pos_idx] - self.rho * step * self.quantile_level
        result[neg_idx] = y[neg_idx] + self.rho * step * (1 - self.quantile_level)

        # Smooth region - use weighted average
        factor = 1 / (1 + step / self.rho)
        result[smooth_idx] = factor * w[smooth_idx] + \
            (1 - factor) * (y[smooth_idx] - shift_cst)

        return result

    def prox_conjugate(self, z, step, y):
        """Prox of step * quantile_huber^*."""
        # Using Moreau decomposition
        inv_step = 1 / step
        return z - step * self.prox(inv_step * z, inv_step, y)

    def raw_grad(self, y, Xw):
        """Compute the gradient of the loss without X."""
        residual = y - Xw
        q = self.quantile_level
        gradient = np.zeros_like(residual)

        # Positive residuals (y > Xw)
        pos_idx = residual > self.rho
        pos_smooth_idx = (0 < residual) & (residual <= self.rho)

        # Negative residuals (y < Xw)
        neg_idx = residual < -self.rho
        neg_smooth_idx = (-self.rho <= residual) & (residual <= 0)

        # Linear regions
        gradient[pos_idx] = -q
        gradient[neg_idx] = -(q - 1)

        # Quadratic transition regions
        gradient[pos_smooth_idx] = -q * residual[pos_smooth_idx] / self.rho
        gradient[neg_smooth_idx] = -(q - 1) * residual[neg_smooth_idx] / self.rho

        return gradient

    def raw_hessian(self, y, Xw):
        """Compute the diagonal of the Hessian."""
        residual = y - Xw
        q = self.quantile_level
        hessian = np.zeros_like(residual)

        # Only in the quadratic regions
        pos_smooth_idx = (0 < residual) & (residual <= self.rho)
        neg_smooth_idx = (-self.rho <= residual) & (residual <= 0)

        hessian[pos_smooth_idx] = q / self.rho
        hessian[neg_smooth_idx] = (q - 1) / self.rho

        return hessian

    def gradient_scalar(self, X, y, w, Xw, j):
        """Compute gradient with respect to feature j."""
        raw_grad_val = self.raw_grad(y, Xw)
        return X[:, j] @ raw_grad_val

    def gradient(self, X, y, Xw):
        """Compute full gradient X.T @ grad."""
        raw_grad_val = self.raw_grad(y, Xw)
        return X.T @ raw_grad_val

    def get_lipschitz(self, X, y):
        """Get coordinate-wise Lipschitz constants."""
        # For Huber, the Lipschitz constant is 1/rho * ||X_j||^2
        return np.sum(X ** 2, axis=0) / self.rho

    def get_global_lipschitz(self, X, y):
        """Get global Lipschitz constant."""
        # Spectral norm approximation
        return np.linalg.norm(X, ord=2) ** 2 / self.rho

    def initialize(self, X, y):
        """Initialize datafit with X and y."""
        pass

    def initialize_sparse(self, X_data, X_indptr, X_indices, y):
        """Initialize datafit with sparse X and y."""
        pass

    def get_spec(self):
        """Get numba spec."""
        spec = (
            ('quantile_level', float64),
            ('rho', float64),
        )
        return spec

    def params_to_dict(self):
        """Get parameters as dict."""
        return dict(quantile_level=self.quantile_level, rho=self.rho)

    def intercept_update_step(self, y, Xw):
        """Calculate the step to update the intercept."""
        return -np.mean(self.raw_grad(y, Xw))


def solve_with_continuation(X, y, datafit, penalty, solver,
                            rho_init=1.0, rho_final=1e-4, n_steps=5):
    """Solve with continuation strategy, gradually reducing smoothness."""
    n_samples, n_features = X.shape

    # Make sure solver has fit_intercept=False
    solver.fit_intercept = False

    w = np.zeros(n_features)
    Xw = np.zeros(n_samples)

    # Logarithmically spaced rho values
    n_steps = 3
    rho_values = np.logspace(np.log10(rho_init), np.log10(rho_final), n_steps)

    for i, rho in enumerate(rho_values):
        print(f"Continuation step {i+1}/{n_steps}, rho={rho:.6f}")
        start_time = time.time()

        # Update rho in the datafit
        datafit.rho = rho

        # Adjust solver parameters based on current rho
        if i < n_steps - 1:
            # Use looser tolerance for intermediate steps
            solver.tol = max(1e-3, rho)
        else:
            # Use tighter tolerance for final step
            solver.tol = 1e-6

        # Solve with current smoothing
        w, _, _ = solver.solve(X, y, datafit, penalty, w_init=w, Xw_init=Xw)
        Xw = X @ w

        print(f"  Step took {time.time() - start_time:.2f} seconds")

    return w


def adaptive_solve(X, y, quantile_level, alpha,
                   rho_init=2.0, rho_final=1e-6, n_steps=8,
                   max_backtrack=2):
    """Choose appropriate solvers based on smoothness level with enhanced stability."""
    import warnings
    from sklearn.exceptions import ConvergenceWarning

    # Initialize datafit and penalty
    datafit = QuantileHuber(quantile_level=quantile_level, rho=rho_init)
    penalty = L1(alpha=alpha)

    # Logarithmically spaced rho values
    rho_values = np.logspace(np.log10(rho_init), np.log10(rho_final), 10)

    n_features = X.shape[1]
    obj_values = []

    # Initialize weights
    w = np.zeros(n_features) + 1e-6
    Xw = np.zeros(X.shape[0])

    # For momentum
    w_prev = w.copy()

    for i, rho in enumerate(rho_values):
        print(f"Continuation step {i+1}/{n_steps}, rho={rho:.6f}")

        # Update rho in the datafit
        datafit.rho = rho

        # Apply momentum for warm starting (except first step)
        if i > 0:
            w_momentum = w + 0.2 * (w - w_prev)
            w_prev = w.copy()
            w = w_momentum
            Xw = X @ w
        else:
            w_prev = w.copy()

        # Select solver based on smoothness
        if i == 0:
            # First step: stabilize with AndersonCD at loose settings
            solver = AndersonCD(tol=1e-2, max_iter=10, fit_intercept=False)
        elif rho > 0.05:
            # Moderately smooth, safe to use ProxNewton now
            solver = ProxNewton(tol=max(1e-3, rho / 5),
                                max_iter=100, fit_intercept=False)
        elif rho > 0.001:
            solver = AndersonCD(tol=max(5e-4, rho / 5),
                                max_iter=50, fit_intercept=False)
        else:
            # Nearly non-smooth — use PDCD_WS
            solver = PDCD_WS(tol=max(1e-4, rho),
                             max_iter=200, max_epochs=200, fit_intercept=False, warm_start=True)

        print(f"  Using solver: {solver.__class__.__name__} for rho={rho:.2e}")

        # Solve with current settings - suppress warnings for intermediate steps
        with warnings.catch_warnings():
            if i < n_steps - 1:
                warnings.filterwarnings("ignore", category=ConvergenceWarning)

            w, p_objs, _ = solver.solve(X, y, datafit, penalty, w_init=w, Xw_init=Xw)

        # Detect oscillation by checking objective values
        if len(p_objs) > 10:
            # Use more robust approach with clipping
            last_values = np.clip(p_objs[-10:], -1e10, 1e10)
            diffs = np.diff(last_values)
            # Avoid overflow in multiplication
            sign_changes = np.sum((diffs[1:] > 0) != (diffs[:-1] > 0))
            if sign_changes > 6:  # Lots of sign changes indicates oscillation
                # Add small random perturbation to escape saddle point
                print("  Detected oscillation, adding perturbation to escape saddle point")
                perturbation = np.random.normal(0, 0.01 * np.std(w), size=w.shape)
                w += perturbation
                Xw = X @ w

        Xw = X @ w

        # Track objective function value
        obj_value = datafit.value(y, w, Xw) + penalty.value(w)
        obj_values.append(obj_value)

        # Backtracking mechanism for numerical stability
        if i > 0 and obj_values[-1] > 1.5 * obj_values[-2]:
            backtrack_count = 0
            while obj_values[-1] > 1.1 * obj_values[-2] and backtrack_count < max_backtrack:
                # Reduce the step size
                backtrack_rho = (rho_values[i-1] * 0.7 + rho) / 2
                print(f"  Backtracking to rho={backtrack_rho:.6f}")

                # Reset to previous weights
                w = w_prev.copy()
                Xw = X @ w

                # Try with intermediate rho
                datafit.rho = backtrack_rho
                w, _, _ = solver.solve(X, y, datafit, penalty, w_init=w, Xw_init=Xw)
                Xw = X @ w

                # Check if improved
                obj_values[-1] = datafit.value(y, w, Xw) + penalty.value(w)
                backtrack_count += 1

        print(f"  Objective value: {obj_values[-1]:.4f}")

    return w


def test_original_issue():
    # Setup the problem that fails
    import numpy as np
    from skglm import GeneralizedLinearEstimator
    from skglm.experimental.pdcd_ws import PDCD_WS
    from skglm.experimental.quantile_regression import Pinball
    from skglm.penalties import L1
    from sklearn.datasets import make_regression
    from sklearn.preprocessing import StandardScaler

    # Generate data
    n_samples, n_features = 1000, 10
    X, y = make_regression(n_samples=n_samples, n_features=n_features,
                           noise=0.1, random_state=42)

    X = StandardScaler().fit_transform(X)
    y = y - y.mean()

    # Setup the model components
    datafit = Pinball(0.5)
    penalty = L1(alpha=0.1)
    solver = PDCD_WS(
        max_iter=500,
        max_epochs=500,
        tol=1e-2,
        warm_start=True,
        verbose=True
    )

    # Try to fit the model
    model = GeneralizedLinearEstimator(datafit=datafit, penalty=penalty, solver=solver)
    try:
        model.fit(X, y)
        print("Original issue: Finished running, but check convergence warnings above.")
    except Exception as e:
        print(f"Original issue reproduced: {e}")

    return X, y


def test_continuation_solution(X, y):
    from skglm.penalties import L1
    from skglm.solvers import AndersonCD

    # Test with your QuantileHuber implementation
    alpha = 0.1
    quantile_level = 0.5

    # Test the adaptive solve approach
    print("\nTesting adaptive_solve:")
    w_adaptive = adaptive_solve(X, y,
                                quantile_level=quantile_level,
                                alpha=alpha,
                                rho_init=1.0,
                                rho_final=1e-6,
                                n_steps=5)

    # Test with a specific solver approach
    print("\nTesting solve_with_continuation:")
    datafit = QuantileHuber(quantile_level=quantile_level, rho=1.0)
    penalty = L1(alpha=alpha)
    solver = AndersonCD(tol=1e-4, max_iter=100, fit_intercept=False)
    w_specific = solve_with_continuation(X, y,
                                         datafit=datafit,
                                         penalty=penalty,
                                         solver=solver,
                                         rho_init=1.0,
                                         rho_final=1e-6,
                                         n_steps=5)

    # Compare with original Pinball if possible
    try:
        from skglm.experimental.quantile_regression import Pinball
        from skglm.solvers import AndersonCD
        from skglm import GeneralizedLinearEstimator

        print("\nTrying standard solver with Pinball for comparison:")
        datafit_pinball = Pinball(quantile_level)
        penalty_pinball = L1(alpha=alpha)
        solver_pinball = AndersonCD(tol=1e-6, max_iter=100)

        model = GeneralizedLinearEstimator(
            datafit=datafit_pinball,
            penalty=penalty_pinball,
            solver=solver_pinball
        )
        model.fit(X, y)
        w_pinball = model.coef_

        # Compare solutions
        print("\nComparing solutions:")
        print(
            f"L2 distance between adaptive and standard: {np.linalg.norm(w_adaptive - w_pinball)}")
        print(
            f"L2 distance between specific and standard: {np.linalg.norm(w_specific - w_pinball)}")

    except Exception as e:
        print(f"Couldn't compare with standard Pinball: {e}")

    # Return solutions for comparison
    return w_adaptive, w_specific


def main():
    X, y = test_original_issue()
    w_adaptive, w_specific = test_continuation_solution(X, y)

    # Print summary
    print("\nSummary:")
    print(f"Adaptive solution first few coefficients: {w_adaptive[:5]}")
    print(f"Specific solution first few coefficients: {w_specific[:5]}")

    # Check solution quality
    from sklearn.metrics import mean_absolute_error

    y_pred_adaptive = X @ w_adaptive
    y_pred_specific = X @ w_specific

    mae_adaptive = mean_absolute_error(y, y_pred_adaptive)
    mae_specific = mean_absolute_error(y, y_pred_specific)

    print(f"MAE of adaptive solution: {mae_adaptive:.4f}")
    print(f"MAE of specific solution: {mae_specific:.4f}")


if __name__ == "__main__":
    main()
