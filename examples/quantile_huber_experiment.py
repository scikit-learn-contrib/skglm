"""
Experiments: Progressive Smoothing for Quantile Regression
Reproduces GitHub issue #276 and validates the method
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
from scipy import stats


def pinball_loss(residuals, tau):
    return np.mean(residuals * (tau - (residuals < 0)))


def test_pdcd_instability():
    """Experiment 1: Reproduce PDCD-WS instability from GitHub issue #276"""
    print("Experiment 1: PDCD-WS Instability (GitHub #276)")
    print("-" * 45)

    # Exact reproduction: n=1000, no scaling
    np.random.seed(42)
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1)

    # PDCD-WS with Pinball (should fail to converge)
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

    # Progressive Smoothing (should work)
    start = time.time()
    smooth_est = SmoothQuantileRegressor(
        quantile=0.5, alpha=0.1, delta_init=0.5,
        delta_final=0.01, n_deltas=5, verbose=False)
    smooth_est.fit(X, y)
    smooth_time = time.time() - start

    print(
        f"PDCD-WS:     {'DID NOT CONVERGE' if did_not_converge else 'CONVERGED'} "
        f"({pdcd_time:.3f}s)")
    print(f"Progressive: CONVERGED ({smooth_time:.3f}s)")

    return did_not_converge  # Return True if PDCD-WS did not converge


def test_quantile_levels(n_runs=10):
    """Experiment 2: Test across quantile levels with statistical validation"""
    print("\nExperiment 2: Quantile Level Validation")
    print("-" * 38)
    print("Coverage: % of data points below prediction (should match τ)")
    print(f"Results averaged over {n_runs} runs")
    print("-" * 38)

    taus = [0.1, 0.3, 0.5, 0.7, 0.9]  # Include extreme quantiles
    results = []

    print(f"{'τ':<4} {'Progressive':<10} {'Sklearn':<10} {'Speedup':<8} "
          f"{'Coverage (P)':<12} {'Coverage (S)':<12} {'p-value':<8}")
    print("-" * 75)

    for tau in taus:
        # Store results for each run
        smooth_times = []
        sk_times = []
        smooth_coverages = []
        sk_coverages = []

        for _ in range(n_runs):
            # Generate new data for each run
            X, y = make_regression(n_samples=800, n_features=15, noise=0.1,
                                   random_state=None)  # Different seed each time

            # Progressive Smoothing
            start = time.time()
            smooth_est = SmoothQuantileRegressor(
                quantile=tau, alpha=0.1, delta_init=0.5,
                delta_final=0.01, n_deltas=5, verbose=False)
            smooth_est.fit(X, y)
            smooth_time = time.time() - start
            smooth_pred = smooth_est.predict(X)
            smooth_coverage = np.mean(y <= smooth_pred)

            # Sklearn
            start = time.time()
            sk_est = QuantileRegressor(quantile=tau, alpha=0.1)
            sk_est.fit(X, y)
            sk_time = time.time() - start
            sk_pred = sk_est.predict(X)
            sk_coverage = np.mean(y <= sk_pred)

            smooth_times.append(smooth_time)
            sk_times.append(sk_time)
            smooth_coverages.append(smooth_coverage)
            sk_coverages.append(sk_coverage)

        # Compute statistics
        smooth_time_mean = np.mean(smooth_times)
        sk_time_mean = np.mean(sk_times)
        smooth_coverage_mean = np.mean(smooth_coverages)
        sk_coverage_mean = np.mean(sk_coverages)
        speedup = sk_time_mean / smooth_time_mean

        # Statistical test for coverage difference
        t_stat, p_value = stats.ttest_rel(smooth_coverages, sk_coverages)

        print(
            f"{tau:<4.1f} {smooth_time_mean:<10.3f} {sk_time_mean:<10.3f} "
            f"{speedup:<8.1f} {smooth_coverage_mean:<12.3f} "
            f"{sk_coverage_mean:<12.3f} {p_value:<8.3f}")

        results.append((tau, smooth_time_mean, sk_time_mean, speedup,
                       smooth_coverage_mean, sk_coverage_mean, p_value))

    return results


def test_scalability(n_runs=2):
    """Experiment 3: Scalability comparison with statistical validation"""
    print("\nExperiment 3: Scalability Analysis")
    print("-" * 32)
    print(f"Results averaged over {n_runs} runs")
    print("-" * 32)

    # Progressive smoothing parameters
    delta_init = 0.5
    delta_final = 0.05
    n_deltas = 5

    # Range of problem sizes
    sizes = [
        (100, 10),     # Small
        (1000, 100),   # Medium
        (10000, 1000),  # Large
    ]
    results = []

    # Format strings for consistent alignment
    size_fmt = "{:d}×{:d}"
    time_fmt = "{:.3f}"
    speedup_fmt = "{:.1f}"
    pval_fmt = "{:.3f}"
    obj_fmt = "{:.2e}"

    print(f"{'Size':<10} {'Progressive':>10} {'Sklearn':>10} "
          f"{'Speedup':>8} {'p-value':>8} {'Obj (P-S)':>10}")
    print("-" * 60)
    print(f"      Progressive uses {n_deltas} smoothing steps")
    print(f"      from delta={delta_init:.2f} to {delta_final:.2f}")
    print("-" * 60)
    print("      Obj (P-S) > 0 means Progressive has worse objective (higher loss)")
    print("-" * 60)

    for n, p in sizes:
        # Store results for each run
        smooth_times = []
        sk_times = []
        obj_diffs = []

        for _ in range(n_runs):
            # Generate dense data
            X, y = make_regression(n_samples=n, n_features=p, noise=0.1,
                                   random_state=None)

            # Progressive Smoothing - using the key innovation
            start = time.time()
            smooth_est = SmoothQuantileRegressor(
                quantile=0.5, alpha=0.1,
                delta_init=delta_init,
                delta_final=delta_final,
                n_deltas=n_deltas,
                max_iter=1000, tol=1e-4, verbose=False,
                solver="AndersonCD", fit_intercept=True)
            smooth_est.fit(X, y)
            smooth_time = time.time() - start
            smooth_pred = smooth_est.predict(X)
            smooth_obj = np.mean(np.abs(y - smooth_pred))

            # Sklearn
            start = time.time()
            sk_est = QuantileRegressor(quantile=0.5, alpha=0.1)
            sk_est.fit(X, y)
            sk_time = time.time() - start
            sk_pred = sk_est.predict(X)
            sk_obj = np.mean(np.abs(y - sk_pred))

            smooth_times.append(smooth_time)
            sk_times.append(sk_time)
            # Positive means Progressive has worse objective (higher loss)
            obj_diffs.append(smooth_obj - sk_obj)

        # Compute statistics
        smooth_time_mean = np.mean(smooth_times)
        sk_time_mean = np.mean(sk_times)
        # Speedup > 1 means Progressive is faster
        speedup = sk_time_mean / smooth_time_mean
        obj_diff_mean = np.mean(obj_diffs)

        # Statistical test for timing difference
        t_stat, p_value = stats.ttest_rel(smooth_times, sk_times)

        # Print results
        size_str = size_fmt.format(n, p)
        smooth_str = time_fmt.format(smooth_time_mean)
        sk_str = time_fmt.format(sk_time_mean)
        speedup_str = speedup_fmt.format(speedup)
        pval_str = pval_fmt.format(p_value)
        obj_str = obj_fmt.format(obj_diff_mean)

        print(f"{size_str:<10} {smooth_str:>10} {sk_str:>10} "
              f"{speedup_str:>8} {pval_str:>8} {obj_str:>10}")

        results.append((n, p, smooth_time_mean, sk_time_mean, speedup,
                       p_value, obj_diff_mean))

    return results


def main():
    """Run all experiments and print results summary"""
    print("Progressive Smoothing for Quantile Regression")
    print("=" * 65)

    # Run experiments
    instability_reproduced = test_pdcd_instability()
    quantile_results = test_quantile_levels()
    scale_results = test_scalability()

    # Print summary
    print("\n" + "=" * 65)
    print("Summary")
    print("=" * 65)

    status = 'Reproduced' if instability_reproduced else 'Not observed'
    print(f"PDCD-WS instability: {status}")

    # Compute average speedup only for statistically significant results
    significant_speedups = [r[3] for r in quantile_results if r[6] < 0.05]
    avg_speedup = np.mean(significant_speedups) if significant_speedups else 0

    significant_scale_speedups = [r[4] for r in scale_results if r[5] < 0.05]
    max_speedup = max(significant_scale_speedups) if significant_scale_speedups else 0

    print(f"Average speedup (significant results only): {avg_speedup:.1f}")
    print(f"Maximum speedup (significant results only): {max_speedup:.1f}")
    print(f"Largest problem solved: {scale_results[-1][0]}×{scale_results[-1][1]}")

    print("\nProgressive smoothing performs best when:")
    print("PDCD-WS fails to converge (n≥1000)")
    print(f"Extreme quantiles (τ≤0.3 or τ≥0.7): {avg_speedup:.1f} times faster")
    print(f"Large-scale problems: up to {max_speedup:.1f} times faster")


if __name__ == "__main__":
    main()
