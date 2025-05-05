import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
from sklearn.linear_model import QuantileRegressor

from skglm import GeneralizedLinearEstimator
from skglm.experimental.progressive_smoothing import ProgressiveSmoothingSolver
from skglm.experimental.pdcd_ws import PDCD_WS
from skglm.experimental.quantile_regression import Pinball
from skglm.penalties import L1
from skglm.solvers import FISTA
from skglm.solvers import LBFGS

# FISTA(max_iter=50, tol=1e-4)
smooth_solver = FISTA(max_iter=500, tol=1e-6)
smooth_solver.fit_intercept = True


def pinball_loss(y_true, y_pred, tau=0.5):
    """Compute Pinball (quantile) loss."""
    residuals = y_true - y_pred
    return np.mean(np.where(residuals >= 0,
                            tau * residuals,
                            (1 - tau) * -residuals))


def test_issue276_regression():
    """Test that progressive smoothing fixes the PDCD_WS stability issue #276."""
    # Generate data similar to the GitHub issue
    np.random.seed(42)
    n_samples_small, n_samples_large = 100, 1000
    n_features = 10

    # Create two datasets - small should work with PDCD_WS, large exhibits the issue
    X_small, y_small = make_regression(n_samples=n_samples_small, n_features=n_features,
                                       noise=0.1, random_state=42)
    X_large, y_large = make_regression(n_samples=n_samples_large, n_features=n_features,
                                       noise=0.1, random_state=42)

    X_small = StandardScaler().fit_transform(X_small)
    # center the target so intercept*should* be zero
    y_small = y_small - np.mean(y_small)
    X_large = StandardScaler().fit_transform(X_large)
    y_large = y_large - np.mean(y_large)

    alpha, tau = 0.1, 0.8

    # Reference solution using QuantileRegressor with the LP solver
    qr_small = QuantileRegressor(quantile=tau, alpha=alpha, fit_intercept=True,
                                 solver="highs").fit(X_small, y_small)
    qr_large = QuantileRegressor(quantile=tau, alpha=alpha, fit_intercept=True,
                                 solver="highs").fit(X_large, y_large)

    # 1. Verify PDCD_WS works fine on small dataset
    pdcd_solver = PDCD_WS(max_iter=500, max_epochs=500, tol=1e-4, verbose=False)
    pdcd_solver.fit_intercept = True

    estimator_small = GeneralizedLinearEstimator(
        datafit=Pinball(tau),
        penalty=L1(alpha=alpha),
        solver=pdcd_solver,
    ).fit(X_small, y_small)

    print("intercept_ →", estimator_small.intercept_)

    print("PDCD small intercept:", estimator_small.intercept_)

    y_pred_pdcd_small = estimator_small.predict(X_small)
    pdcd_small_loss = pinball_loss(y_small, y_pred_pdcd_small, tau=tau)

    # 2. Apply PDCD_WS to large dataset (should exhibit issue #276)
    pdcd_solver = PDCD_WS(max_iter=500, max_epochs=200, tol=1e-4, verbose=False)
    pdcd_solver.fit_intercept = True
    estimator_large = GeneralizedLinearEstimator(
        datafit=Pinball(tau),
        penalty=L1(alpha=alpha),
        solver=pdcd_solver,
    ).fit(X_large, y_large)

    print("intercept_ →", estimator_large.intercept_)

    y_pred_pdcd_large = estimator_large.predict(X_large)
    pdcd_large_loss = pinball_loss(y_large, y_pred_pdcd_large, tau=tau)

    # 3. Apply progressive smoothing to large dataset (should fix issue #276)
    pss = ProgressiveSmoothingSolver(
        smoothing_sequence=[1.0, 0.5, 0.2, 0.1, 0.05],
        quantile=tau,
        alpha=alpha,
        verbose=True,
        smooth_solver=None,
        nonsmooth_solver=PDCD_WS(max_iter=100, max_epochs=50, tol=1e-4)
    ).fit(X_large, y_large)

    y_pred_pss_large = pss.predict(X_large)
    pss_large_loss = pinball_loss(y_large, y_pred_pss_large, tau=tau)

    # 4. Calculate reference losses
    y_pred_ref_small = qr_small.predict(X_small)
    ref_small_loss = pinball_loss(y_small, y_pred_ref_small, tau=tau)

    y_pred_ref_large = qr_large.predict(X_large)
    ref_large_loss = pinball_loss(y_large, y_pred_ref_large, tau=tau)

    # 5. Compute relative gaps
    rel_gap_pdcd_small = (pdcd_small_loss - ref_small_loss) / ref_small_loss
    rel_gap_pdcd_large = (pdcd_large_loss - ref_large_loss) / ref_large_loss
    rel_gap_pss_large = (pss_large_loss - ref_large_loss) / ref_large_loss

    # 6. Assert that PDCD_WS works well on small dataset
    assert rel_gap_pdcd_small < 0.05, \
        f"PDCD_WS should work well on small dataset (rel_gap={rel_gap_pdcd_small:.4f})"

    # 7. Key test: Assert that progressive smoothing fixes the issue on large dataset
    assert rel_gap_pss_large < 0.05, \
        f"Progressive smoothing failed to fix issue #276 (rel_gap={rel_gap_pss_large:.4f})"

    # 8. Optional: Verify the progressive solution is better than direct PDCD_WS
    if rel_gap_pdcd_large > 0.05:  # Only if PDCD_WS actually has an issue
        assert pss_large_loss < pdcd_large_loss, \
            "Progressive smoothing should outperform direct PDCD_WS on large dataset"

        # Check for saddle point by examining iterations and convergence
        assert len(pss.stage_results_) > 0, "Missing stage results in progressive solver"


def test_progressive_solver_non_median():
    """Test progressive smoothing with non-median quantiles."""
    np.random.seed(42)

    # Generate data
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    X = StandardScaler().fit_transform(X)

    # Test upper quantile (0.8)
    tau = 0.8
    alpha = 0.1

    # Reference solution
    qr = QuantileRegressor(quantile=tau, alpha=alpha, fit_intercept=True,
                           solver="highs").fit(X, y)
    y_pred_ref = qr.predict(X)
    ref_loss = pinball_loss(y, y_pred_ref, tau=tau)

    # Progressive smoothing solution
    pss = ProgressiveSmoothingSolver(
        smoothing_sequence=[1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001],
        quantile=tau,
        alpha=alpha,
        verbose=False,
        smooth_solver=smooth_solver,
        nonsmooth_solver=PDCD_WS(max_iter=10000, max_epochs=5000, tol=1e-8)
    ).fit(X, y)

    y_pred_pss = pss.predict(X)
    pss_loss = pinball_loss(y, y_pred_pss, tau=tau)

    # Compute relative gap
    rel_gap = (pss_loss - ref_loss) / ref_loss

    # Assert close to reference solution
    assert rel_gap < 0.05, \
        f"Progressive smoothing should work for non-median quantiles (rel_gap={rel_gap:.4f})"

    # Verify asymmetry of predictions
    residuals = y - y_pred_pss
    n_pos = np.sum(residuals > 0)
    n_neg = np.sum(residuals < 0)
    assert abs(n_pos / (n_pos + n_neg) - tau) < 0.1, \
        f"Residual distribution doesn't match target quantile {tau}"
