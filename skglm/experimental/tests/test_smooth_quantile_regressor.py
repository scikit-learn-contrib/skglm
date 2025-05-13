import numpy as np
import time
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


def test_pdcd_ws_vs_smooth_quantile():
    """
    Compare PDCD_WS and SmoothQuantileRegressor on small and large datasets.
    PDCD_WS should work on small data, struggle on large;
    SmoothQuantileRegressor should be robust.
    """
    np.random.seed(42)
    n_small, n_large = 100, 1000
    n_features = 10
    Xs, ys = [], []
    for n in [n_small, n_large]:
        X, y = make_regression(n_samples=n, n_features=n_features,
                               noise=0.1, random_state=42)
        X = StandardScaler().fit_transform(X)
        y = y - np.mean(y)
        Xs.append(X)
        ys.append(y)
    X_small, y_small = Xs[0], ys[0]
    X_large, y_large = Xs[1], ys[1]
    alpha, tau = 0.1, 0.5
    # Reference solutions
    qr_small = QuantileRegressor(quantile=tau, alpha=alpha, fit_intercept=True,
                                 solver="highs").fit(X_small, y_small)
    qr_large = QuantileRegressor(quantile=tau, alpha=alpha, fit_intercept=True,
                                 solver="highs").fit(X_large, y_large)
    # PDCD_WS small
    pdcd_solver = PDCD_WS(max_iter=500, max_epochs=500, tol=1e-4, verbose=False)
    pdcd_solver.fit_intercept = True
    est_small = GeneralizedLinearEstimator(
        datafit=Pinball(tau), penalty=L1(alpha=alpha), solver=pdcd_solver
    ).fit(X_small, y_small)
    y_pred_pdcd_small = est_small.predict(X_small)
    pdcd_small_loss = pinball_loss(y_small, y_pred_pdcd_small, tau=tau)
    # PDCD_WS large
    pdcd_solver = PDCD_WS(max_iter=500, max_epochs=200, tol=1e-4, verbose=False)
    pdcd_solver.fit_intercept = True
    est_large = GeneralizedLinearEstimator(
        datafit=Pinball(tau), penalty=L1(alpha=alpha), solver=pdcd_solver
    ).fit(X_large, y_large)
    y_pred_pdcd_large = est_large.predict(X_large)
    pdcd_large_loss = pinball_loss(y_large, y_pred_pdcd_large, tau=tau)
    # SmoothQuantileRegressor large
    sqr = SmoothQuantileRegressor(
        smoothing_sequence=[1.0, 0.5, 0.3, 0.2, 0.15, 0.12, 0.1, 0.08, 0.06,
                            0.04, 0.02, 0.01, 0.005, 0.001],
        quantile=tau, alpha=alpha, verbose=False
    ).fit(X_large, y_large)
    y_pred_sqr_large = sqr.predict(X_large)
    sqr_large_loss = pinball_loss(y_large, y_pred_sqr_large, tau=tau)
    # Reference losses
    ref_small_loss = pinball_loss(y_small, qr_small.predict(X_small), tau=tau)
    ref_large_loss = pinball_loss(y_large, qr_large.predict(X_large), tau=tau)
    # Relative gaps
    rel_gap_pdcd_small = (pdcd_small_loss - ref_small_loss) / ref_small_loss
    rel_gap_pdcd_large = (pdcd_large_loss - ref_large_loss) / ref_large_loss
    rel_gap_sqr_large = (sqr_large_loss - ref_large_loss) / ref_large_loss
    assert rel_gap_pdcd_small < 0.05, (
        "PDCD_WS should work well on small dataset (rel_gap={:.4f})"
        .format(rel_gap_pdcd_small))
    assert rel_gap_sqr_large < 0.05, (
        "SmoothQuantileRegressor failed to fix large dataset (rel_gap={:.4f})"
        .format(rel_gap_sqr_large))
    if rel_gap_pdcd_large > 0.05:
        assert sqr_large_loss < pdcd_large_loss, (
            "SmoothQuantileRegressor should outperform PDCD_WS on large dataset"
        )
        assert len(sqr.stage_results_) > 0, (
            "Missing stage results in SmoothQuantileRegressor"
        )


def test_smooth_quantile_nonmedian():
    """
    Test SmoothQuantileRegressor for non-median quantiles (e.g., tau=0.8).
    """
    np.random.seed(42)
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    X = StandardScaler().fit_transform(X)
    tau = 0.8
    alpha = 0.1
    qr = QuantileRegressor(quantile=tau, alpha=alpha,
                           fit_intercept=True, solver="highs").fit(X, y)
    y_pred_ref = qr.predict(X)
    ref_loss = pinball_loss(y, y_pred_ref, tau=tau)
    sqr = SmoothQuantileRegressor(
        smoothing_sequence=[1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001],
        quantile=tau, alpha=alpha, verbose=False, smooth_solver=smooth_solver).fit(X, y)
    y_pred_sqr = sqr.predict(X)
    sqr_loss = pinball_loss(y, y_pred_sqr, tau=tau)
    rel_gap = (sqr_loss - ref_loss) / ref_loss
    assert rel_gap < 0.05, (
        "SmoothQuantileRegressor should work for non-median quantiles "
        "(rel_gap={:.4f})".format(rel_gap)
    )
    residuals = y - y_pred_sqr
    n_pos = np.sum(residuals > 0)
    n_neg = np.sum(residuals < 0)
    assert abs(n_pos / (n_pos + n_neg) -
               tau) < 0.1, (
        f"Residual distribution doesn't match target quantile {tau}")


def test_smooth_quantile_performance():
    """
    Test SmoothQuantileRegressor performance and accuracy vs scikit-learn on large data.
    """
    n_samples = 1000
    n_features = 10
    tau = 0.5
    alpha = 0.1
    rel_gap_tol = 0.05
    X, y = make_regression(n_samples=n_samples, n_features=n_features,
                           noise=0.1, random_state=42)
    X = StandardScaler().fit_transform(X)
    y = y - np.mean(y)

    # scikit-learn's QuantileRegressor
    start_time = time.time()
    qr = QuantileRegressor(quantile=tau, alpha=alpha, fit_intercept=True,
                           solver="highs").fit(X, y)
    qr_time = time.time() - start_time
    y_pred_qr = qr.predict(X)
    qr_loss = pinball_loss(y, y_pred_qr, tau=tau)

    # SmoothQuantileRegressor
    start_time = time.time()
    sqr = SmoothQuantileRegressor(
        smoothing_sequence=[1.0, 0.5, 0.2, 0.1, 0.05],
        quantile=tau, alpha=alpha, verbose=False,
        smooth_solver=FISTA(max_iter=2000, tol=1e-8)
    ).fit(X, y)
    sqr_time = time.time() - start_time
    y_pred_sqr = sqr.predict(X)
    sqr_loss = pinball_loss(y, y_pred_sqr, tau=tau)

    # Calculate metrics
    rel_gap = (sqr_loss - qr_loss) / qr_loss
    speedup = qr_time / sqr_time

    # Print performance comparison
    print("\nPerformance Comparison:")
    print("scikit-learn QuantileRegressor:")
    print(f"  Time: {qr_time:.3f}s")
    print(f"  Loss: {qr_loss:.6f}")
    print("SmoothQuantileRegressor:")
    print(f"  Time: {sqr_time:.3f}s")
    print(f"  Loss: {sqr_loss:.6f}")
    print(f"  Speedup: {speedup:.1f}x")
    print(f"  Relative gap: {rel_gap:.1%}")

    # Assertions
    assert rel_gap < rel_gap_tol, (
        "SmoothQuantileRegressor should achieve similar accuracy to "
        "scikit-learn's QuantileRegressor (rel_gap={:.4f})".format(rel_gap)
    )
    assert sqr_time < qr_time, (
        f"SmoothQuantileRegressor should be faster than scikit-learn's "
        f"QuantileRegressor (sqr_time={sqr_time:.2f}s vs qr_time={qr_time:.2f}s)"
    )
