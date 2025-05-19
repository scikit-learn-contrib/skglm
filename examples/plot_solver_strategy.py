"""
======================================
Staged Optimization with Dynamic Solvers
======================================

This example demonstrates how the StageBasedSolverStrategy can be used
to progressively solve regularization paths with adaptive solver configurations.

Compare three ways to compute a regularisation path for a synthetic
least‑squares problem:

1. **Standard** – solve each alpha from scratch with `PDCD_WS`.
2. **Staged strategy** – warm‑start `PDCD_WS` and let
   `StageBasedSolverStrategy` tighten *tol* and increase `max_iter`
   stage by stage.
3. **LARS** – scikit‑learn’s specialised path algorithm (reference speed).

The script generates four panels:

* prediction error vs alpha
* model sparsity vs alpha
* time per stage (standard vs strategy)
* the strategy’s internal *tol* / `max_iter` schedule

Run::

    python plot_solver_strategy.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from skglm.experimental.pdcd_ws import PDCD_WS
from skglm.experimental.solver_strategies import StageBasedSolverStrategy
import time
from sklearn.linear_model import lars_path
from skglm.experimental.quantile_huber import QuantileHuber  # Compatible with PDCD_WS
from skglm.penalties import L1
from skglm.estimators import GeneralizedLinearEstimator

np.random.seed(42)

n_samples, n_features = 200, 500
n_informative = 20

# Create data with a structured sparsity pattern
X, y, true_coef = make_regression(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=n_informative,
    noise=0.5,
    coef=True,
    random_state=42
)
X = StandardScaler().fit_transform(X)

# Create a geometric sequence of regularization parameters (alpha values)
alpha_max = np.abs(X.T @ y).max() / n_samples
alpha_min = alpha_max / 100
n_alphas = 10
alphas = np.geomspace(alpha_max, alpha_min, n_alphas)

# Method 1: Standard Approach - Independent Problems
standard_results = []
standard_time_total = 0
standard_coef_path = []

for i, alpha in enumerate(alphas):
    # QuantileHuber with quantile=0.5 acts like a Huber loss, which works with PDCD_WS
    datafit = QuantileHuber(delta=0.1, quantile=0.5)
    solver = PDCD_WS(
        max_iter=1000,
        tol=1e-5,
        fit_intercept=True
    )

    # Create estimator with compatible datafit
    estimator = GeneralizedLinearEstimator(
        datafit=datafit,
        penalty=L1(alpha=alpha),
        solver=solver
    )

    start_time = time.time()
    estimator.fit(X, y)
    elapsed = time.time() - start_time
    standard_time_total += elapsed

    y_pred = estimator.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    nnz = np.count_nonzero(estimator.coef_)

    standard_coef_path.append(estimator.coef_.copy())
    standard_results.append({
        'alpha': alpha, 'mse': mse, 'r2': r2,
        'nnz': nnz, 'time': elapsed
    })

# Method 2: Continuation Path with StageBasedSolverStrategy

# Create the base solver
base_solver = PDCD_WS(max_iter=100, tol=1e-4, fit_intercept=True)

# Create the strategy with custom configuration
solver_strategy = StageBasedSolverStrategy({
    'base_tol': 1e-5,
    'tol_delta_factor': 1e-3,
    'max_iter_start': 150,
    'max_iter_step': 100,
    'p0_frac_large': 0.1,
})

# Tracking metrics
strategy_results = []
strategy_time_total = 0
strategy_coef_path = []

# Use the strategy to solve a sequence of Lasso problems
coefs = None

for stage, alpha in enumerate(alphas):
    # Get a solver configured for this stage
    solver = solver_strategy.create_solver_for_stage(
        base_solver, delta=alpha, stage=stage, n_features=n_features
    )

    # Create estimator with compatible datafit
    datafit = QuantileHuber(delta=0.1, quantile=0.5)
    estimator = GeneralizedLinearEstimator(
        datafit=datafit,
        penalty=L1(alpha=alpha),
        solver=solver
    )
    estimator.intercept_ = 0.0  # Initialize intercept_ attribute

    # Warm start from previous solution if available
    if coefs is not None:
        estimator.coef_ = coefs

    # Time the fitting
    start_time = time.time()
    estimator.fit(X, y)
    elapsed = time.time() - start_time
    strategy_time_total += elapsed

    # Save results
    coefs = estimator.coef_.copy()
    strategy_coef_path.append(coefs.copy())
    y_pred = estimator.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    nnz = np.count_nonzero(coefs)

    strategy_results.append({
        'alpha': alpha, 'mse': mse, 'r2': r2,
        'nnz': nnz, 'time': elapsed, 'stage': stage
    })

    # Early stopping if we have more non-zeros than informative features
    if nnz > 2 * n_informative and stage > 2:
        break

# Method 3: scikit-learn's LARS Path Implementation

start_time = time.time()
alphas_lars, _, coefs_lars = lars_path(X, y, method='lasso', alpha_min=alpha_min)
lars_time = time.time() - start_time


# Performance Comparison
speedup_vs_standard = standard_time_total / strategy_time_total
speedup_vs_lars = lars_time / strategy_time_total

print("\nPerformance Comparison:")
print("=" * 60)
print(f"Method 1 (Standard): Total time = {standard_time_total:.3f}s")
print(f"Method 2 (Strategy): Total time = {strategy_time_total:.3f}s")
print(f"Method 3 (LARS): Total time = {lars_time:.3f}s")
print(f"Speedup of Strategy vs Standard: {speedup_vs_standard:.1f}x")
if speedup_vs_lars > 1:
    print(f"Speedup of Strategy vs LARS: {speedup_vs_lars:.1f}x")
else:
    print(f"LARS is {1/speedup_vs_lars:.1f}x faster than Strategy")

# Create figure with 2x2 subplots
fig, axarr = plt.subplots(2, 2, figsize=(12, 9))

# Plot 1: MSE vs Alpha
ax1 = axarr[0, 0]
ax1.plot([r['alpha'] for r in standard_results],
         [r['mse'] for r in standard_results],
         'o-', label='Standard')
ax1.plot([r['alpha'] for r in strategy_results],
         [r['mse'] for r in strategy_results],
         's-', label='Strategy')
ax1.set_xscale('log')
ax1.set_xlabel(r'Regularization parameter $\alpha$')
ax1.set_ylabel('Mean Squared Error')
ax1.set_title('Prediction Error vs. Regularization')
ax1.legend()
ax1.grid(True)

# Plot 2: Non-zeros vs Alpha
ax2 = axarr[0, 1]
ax2.plot([r['alpha'] for r in standard_results],
         [r['nnz'] for r in standard_results],
         'o-', label='Standard')
ax2.plot([r['alpha'] for r in strategy_results],
         [r['nnz'] for r in strategy_results],
         's-', label='Strategy')
ax2.axhline(y=n_informative, linestyle='--', color='r',
            label=f'True non-zeros: {n_informative}')
ax2.set_xscale('log')
ax2.set_xlabel(r'Regularization parameter $\alpha$')
ax2.set_ylabel('Number of non-zero coefficients')
ax2.set_title('Model Sparsity vs. Regularization')
ax2.legend()
ax2.grid(True)

# Plot 3: Solution time per stage comparison
ax3 = axarr[1, 0]
bar_width = 0.35
indices = np.arange(min(len(standard_results), len(strategy_results)))

standard_times = [r['time'] for r in standard_results][:len(indices)]
strategy_times = [r['time'] for r in strategy_results][:len(indices)]

ax3.bar(indices - bar_width/2, standard_times,
        bar_width, label='Standard', color='#1f77b4')
ax3.bar(indices + bar_width/2, strategy_times,
        bar_width, label='Strategy', color='#ff7f0e')
ax3.set_xlabel('Stage')
ax3.set_ylabel('Time (seconds)')
ax3.set_title('Solution Time per Stage')
ax3.set_xticks(indices)
ax3.set_xticklabels([f'{i+1}' for i in indices])
ax3.legend()
ax3.grid(True)

# Plot 4: Solver parameter adaptation
ax4 = axarr[1, 1]
ax4.set_title('Solver Parameter Adaptation')
ax4.set_xlabel('Stage')
ax4.set_ylabel('Tolerance (log scale)', color='blue')
ax4.semilogy([r['stage'] for r in strategy_results],
             [solver_strategy.config['tol_delta_factor'] * r['alpha']
              for r in strategy_results],
             'o-', color='blue', label='Tolerance')
ax4.tick_params(axis='y', labelcolor='blue')

ax4_twin = ax4.twinx()
ax4_twin.set_ylabel('Max iterations', color='red')
ax4_twin.plot([r['stage'] for r in strategy_results],
              [solver_strategy.config['max_iter_start'] +
               solver_strategy.config['max_iter_step'] * r['stage']
               for r in strategy_results],
              's-', color='red', label='Max iterations')
ax4_twin.tick_params(axis='y', labelcolor='red')

lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4_twin.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()
plt.show()
