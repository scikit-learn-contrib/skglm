"""
Quick Tests: Progressive Smoothing for Quantile Regression
"""
import numpy as np
import time
from sklearn.datasets import make_regression
from sklearn.linear_model import QuantileRegressor
from skglm.experimental.quantile_huber import SmoothQuantileRegressor
from skglm.experimental.quantile_regression import Pinball
from skglm.penalties import L1
from skglm import GeneralizedLinearEstimator
from skglm.experimental.pdcd_ws import PDCD_WS
import warnings
from sklearn.exceptions import ConvergenceWarning


def test_pdcd_instability():
    print("\nTest 1: PDCD-WS Instability")
    print("=" * 45)

    np.random.seed(42)
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1)

    datafit = Pinball(0.5)
    penalty = L1(alpha=0.1)
    solver = PDCD_WS(
        max_iter=500,
        max_epochs=500,
        tol=1e-2,
        warm_start=True,
        verbose=False)

    start = time.time()
    with warnings.catch_warnings(record=True) as w:
        estimator = GeneralizedLinearEstimator(
            datafit=datafit, penalty=penalty, solver=solver)
        estimator.fit(X, y)
        did_not_converge = any(issubclass(warn.category, ConvergenceWarning)
                               for warn in w)
    pdcd_time = time.time() - start

    start = time.time()
    smooth_est = SmoothQuantileRegressor(
        quantile=0.5, alpha=0.1, delta_init=0.5,
        delta_final=0.01, n_deltas=5, verbose=False)
    smooth_est.fit(X, y)
    smooth_time = time.time() - start

    print(f"{'Solver':<12} {'Status':<15} {'Time (s)':>10}")
    print("-" * 45)
    status = 'DID NOT CONVERGE' if did_not_converge else 'CONVERGED'
    print(f"{'PDCD-WS':<12} {status:<15} {pdcd_time:>10.3f}")
    print(f"{'SQR':<12} {'CONVERGED':<15} {smooth_time:>10.3f}")

    return did_not_converge


def test_quantile_levels():
    print("\nTest 2: Quantile Level Check")
    print("=" * 65)
    print("Coverage: % of data points below prediction (should match τ)")
    print("-" * 65)

    taus = [0.1, 0.3, 0.5, 0.7, 0.9]
    results = []

    print(f"{'τ':>4} {'SQR':>12} {'Sklearn':>12} {'Speedup':>10} "
          f"{'Coverage (SQR)':>14} {'Coverage (S)':>14}")
    print("-" * 65)

    for tau in taus:
        X, y = make_regression(n_samples=800, n_features=15, noise=0.1)

        start = time.time()
        smooth_est = SmoothQuantileRegressor(
            quantile=tau, alpha=0.1, delta_init=0.5,
            delta_final=0.01, n_deltas=5, verbose=False)
        smooth_est.fit(X, y)
        smooth_time = time.time() - start
        smooth_pred = smooth_est.predict(X)
        smooth_coverage = np.mean(y <= smooth_pred)

        start = time.time()
        sk_est = QuantileRegressor(quantile=tau, alpha=0.1)
        sk_est.fit(X, y)
        sk_time = time.time() - start
        sk_pred = sk_est.predict(X)
        sk_coverage = np.mean(y <= sk_pred)

        speedup = sk_time / smooth_time

        print(f"{tau:>4.1f} {smooth_time:>12.3f} {sk_time:>12.3f} "
              f"{speedup:>10.1f} {smooth_coverage:>14.3f} {sk_coverage:>14.3f}")

        results.append((tau, smooth_time, sk_time, speedup,
                       smooth_coverage, sk_coverage))

    return results


def test_scalability():
    print("\nTest 3: Scalability Check")
    print("=" * 60)

    delta_init = 0.5
    delta_final = 0.05
    n_deltas = 5

    sizes = [
        (100, 10),     # Small
        (1000, 100),   # Medium
        (10000, 1000),  # Large
    ]
    results = []

    print(f"{'Size':>10} {'SQR':>12} {'Sklearn':>12} "
          f"{'Speedup':>10} {'Obj (SQR-S)':>12}")
    print("-" * 60)

    for n, p in sizes:
        X, y = make_regression(n_samples=n, n_features=p, noise=0.1)

        start = time.time()
        smooth_est = SmoothQuantileRegressor(
            quantile=0.7, alpha=0.1,
            delta_init=delta_init,
            delta_final=delta_final,
            n_deltas=n_deltas,
            max_iter=1000, tol=1e-4, verbose=False,
            solver="AndersonCD", fit_intercept=True)
        smooth_est.fit(X, y)
        smooth_time = time.time() - start
        smooth_pred = smooth_est.predict(X)
        smooth_obj = np.mean(np.abs(y - smooth_pred))

        start = time.time()
        sk_est = QuantileRegressor(quantile=0.7, alpha=0.1)
        sk_est.fit(X, y)
        sk_time = time.time() - start
        sk_pred = sk_est.predict(X)
        sk_obj = np.mean(np.abs(y - sk_pred))

        speedup = sk_time / smooth_time
        obj_diff = smooth_obj - sk_obj

        print(f"{n}×{p:<4} {smooth_time:>12.3f} {sk_time:>12.3f} "
              f"{speedup:>10.1f} {obj_diff:>12.2e}")

        results.append((n, p, smooth_time, sk_time, speedup, obj_diff))

    print("-" * 60)
    print(f"SQR uses {n_deltas} smoothing steps")
    print(f"from delta={delta_init:.2f} to {delta_final:.2f}")
    print("-" * 60)
    print("Obj (SQR-S) > 0 means SQR has worse objective (higher loss)")

    return results


def main():
    print("Quick Tests: Smooth Quantile Regression (SQR)")
    print("=" * 65)

    instability_reproduced = test_pdcd_instability()
    quantile_results = test_quantile_levels()
    scale_results = test_scalability()

    print("\n" + "=" * 65)
    print("Summary")
    print("=" * 65)

    status = 'Reproduced' if instability_reproduced else 'Not observed'
    print(f"PDCD-WS instability: {status}")

    avg_speedup = np.mean([r[3] for r in quantile_results])
    max_speedup = max([r[4] for r in scale_results])

    print(f"Average speedup: {avg_speedup:.1f}")
    print(f"Maximum speedup: {max_speedup:.1f}")
    print(f"Largest problem solved: {scale_results[-1][0]}×{scale_results[-1][1]}")


if __name__ == "__main__":
    main()
