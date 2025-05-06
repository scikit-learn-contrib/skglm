import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
from sklearn.linear_model import QuantileRegressor

from skglm import GeneralizedLinearEstimator
from skglm.experimental.smooth_quantile_regressor import SmoothQuantileRegressor
from skglm.experimental.pdcd_ws import PDCD_WS
from skglm.experimental.quantile_regression import Pinball
from skglm.penalties import L1
from skglm.solvers import FISTA

# Default solver configuration for smooth optimization
smooth_solver = FISTA(max_iter=500, tol=1e-6)
smooth_solver.fit_intercept = True


def pinball_loss(y_true, y_pred, tau=0.5):
    """
    Compute Pinball (quantile) loss."""
    residuals = y_true - y_pred
    return np.mean(np.where(residuals >= 0,
                            tau * residuals,
                            (1 - tau) * -residuals))


def test_issue276_regression():
    """
    Test that SmoothQuantileRegressor fixes the PDCD_WS stability issue #276.

    This test demonstrates that:
    1. PDCD_WS works well on small datasets
    2. PDCD_WS struggles on large datasets
    3. SmoothQuantileRegressor solves the issue on large datasets
    """
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
    y_small = y_small - np.mean(y_small)
    X_large = StandardScaler().fit_transform(X_large)
    y_large = y_large - np.mean(y_large)

    alpha, tau = 0.1, 0.5

    # Reference solution using QuantileRegressor with the LP solver
    qr_small = QuantileRegressor(quantile=tau, alpha=alpha, fit_intercept=True,
                                 solver="highs").fit(X_small, y_small)
    qr_large = QuantileRegressor(quantile=tau, alpha=alpha, fit_intercept=True,
                                 solver="highs").fit(X_large, y_large)

    # Verify PDCD_WS works fine on small dataset
    pdcd_solver = PDCD_WS(max_iter=500, max_epochs=500, tol=1e-4, verbose=False)
    pdcd_solver.fit_intercept = True

    estimator_small = GeneralizedLinearEstimator(
        datafit=Pinball(tau),
        penalty=L1(alpha=alpha),
        solver=pdcd_solver,
    ).fit(X_small, y_small)

    y_pred_pdcd_small = estimator_small.predict(X_small)
    pdcd_small_loss = pinball_loss(y_small, y_pred_pdcd_small, tau=tau)

    # Apply PDCD_WS to large dataset (should exhibit issue #276)
    pdcd_solver = PDCD_WS(max_iter=500, max_epochs=200, tol=1e-4, verbose=False)
    pdcd_solver.fit_intercept = True
    estimator_large = GeneralizedLinearEstimator(
        datafit=Pinball(tau),
        penalty=L1(alpha=alpha),
        solver=pdcd_solver,
    ).fit(X_large, y_large)

    y_pred_pdcd_large = estimator_large.predict(X_large)
    pdcd_large_loss = pinball_loss(y_large, y_pred_pdcd_large, tau=tau)

    # Apply SmoothQuantileRegressor to large dataset (should fix issue #276)
    sqr = SmoothQuantileRegressor(
        smoothing_sequence=[1.0, 0.5, 0.3, 0.2, 0.15, 0.12,
                            0.1, 0.08, 0.06, 0.04, 0.02, 0.01, 0.005, 0.001],
        quantile=tau,
        alpha=alpha,
        verbose=False,
    ).fit(X_large, y_large)

    y_pred_sqr_large = sqr.predict(X_large)
    sqr_large_loss = pinball_loss(y_large, y_pred_sqr_large, tau=tau)

    # Reference losses
    y_pred_ref_small = qr_small.predict(X_small)
    ref_small_loss = pinball_loss(y_small, y_pred_ref_small, tau=tau)

    y_pred_ref_large = qr_large.predict(X_large)
    ref_large_loss = pinball_loss(y_large, y_pred_ref_large, tau=tau)

    # Relative gaps
    rel_gap_pdcd_small = (pdcd_small_loss - ref_small_loss) / ref_small_loss
    rel_gap_pdcd_large = (pdcd_large_loss - ref_large_loss) / ref_large_loss
    rel_gap_sqr_large = (sqr_large_loss - ref_large_loss) / ref_large_loss

    assert rel_gap_pdcd_small < 0.05, \
        f"PDCD_WS should work well on small dataset" \
        f"(rel_gap={rel_gap_pdcd_small:.4f})"

    assert rel_gap_sqr_large < 0.05, \
        f"SmoothQuantileRegressor failed to fix issue #276" \
        f"(rel_gap={rel_gap_sqr_large:.4f})"

    if rel_gap_pdcd_large > 0.05:
        assert sqr_large_loss < pdcd_large_loss, \
            "SmoothQuantileRegressor should outperform direct" \
            "PDCD_WS on large dataset"

        assert len(sqr.stage_results_) > 0, "Missing stage results" \
            "in SmoothQuantileRegressor"

    return rel_gap_pdcd_small, rel_gap_pdcd_large, rel_gap_sqr_large


def test_smooth_quantile_regressor_non_median():
    """
    Test SmoothQuantileRegressor with non-median quantiles.

    Verifies that:
    1. The regressor works for tau=0.8 (upper quantile)
    2. The resulting residuals match the expected distribution
    """
    np.random.seed(42)

    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    X = StandardScaler().fit_transform(X)

    tau = 0.8
    alpha = 0.1

    # Reference solution
    qr = QuantileRegressor(quantile=tau, alpha=alpha, fit_intercept=True,
                           solver="highs").fit(X, y)
    y_pred_ref = qr.predict(X)
    ref_loss = pinball_loss(y, y_pred_ref, tau=tau)

    # SmoothQuantileRegressor solution
    sqr = SmoothQuantileRegressor(
        smoothing_sequence=[1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001],
        quantile=tau,
        alpha=alpha,
        verbose=False,
        smooth_solver=smooth_solver,
    ).fit(X, y)

    y_pred_sqr = sqr.predict(X)
    sqr_loss = pinball_loss(y, y_pred_sqr, tau=tau)

    rel_gap = (sqr_loss - ref_loss) / ref_loss

    assert rel_gap < 0.05, \
        f"SmoothQuantileRegressor should work for non-median quantiles" \
        f"(rel_gap={rel_gap:.4f})"

    residuals = y - y_pred_sqr
    n_pos = np.sum(residuals > 0)
    n_neg = np.sum(residuals < 0)
    assert abs(n_pos / (n_pos + n_neg) - tau) < 0.1, \
        f"Residual distribution doesn't match target quantile {tau}"

    return rel_gap
