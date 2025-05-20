"""
===========================================
Fast Quantile Regression with Smoothing
===========================================

NOTE: FOR NOW, SMOOTH QUANTILE IS NOT YET FASTER THAN QUANTILE REGRESSOR.
"""

# %%
# Understanding Progressive Smoothing
# ----------------------------------
#
# The SmoothQuantileRegressor uses a progressive smoothing approach to solve
# quantile regression problems. It starts with a highly smoothed approximation
# and gradually reduces the smoothing parameter to approach the original
# non-smooth problem. This approach is particularly effective for large datasets
# where direct optimization of the non-smooth objective can be challenging.

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import QuantileRegressor
from skglm.experimental.smooth_quantile_regressor import SmoothQuantileRegressor
from skglm.experimental.pdcd_ws import PDCD_WS
from sklearn.model_selection import train_test_split
import pandas as pd
from scipy import stats

# %%
# Data Generation
# --------------
#
# We'll generate synthetic data with different noise distributions to test
# the robustness of our approach. This includes:
# - Exponential noise: Heavy-tailed distribution
# - Student's t noise: Heavy-tailed with controlled degrees of freedom
# - Mixture noise: Combination of normal and exponential distributions


def generate_data(n_samples, n_features, noise_type='exponential', random_state=42):
    """Generate data with different noise distributions."""
    np.random.seed(random_state)
    X, y_base = make_regression(n_samples=n_samples, n_features=n_features,
                                noise=0.1, random_state=random_state)
    X = StandardScaler().fit_transform(X)
    y_base = y_base - np.mean(y_base)  # Center y

    if noise_type == 'exponential':
        noise = np.random.exponential(scale=1.0, size=n_samples) - 1.0
    elif noise_type == 'student_t':
        noise = stats.t.rvs(df=3, size=n_samples)  # Heavy-tailed
    elif noise_type == 'mixture':
        # Mixture of normal and exponential
        mask = np.random.random(n_samples) < 0.7
        noise = np.zeros(n_samples)
        noise[mask] = np.random.normal(0, 0.5, size=mask.sum())
        noise[~mask] = np.random.exponential(scale=1.0, size=(~mask).sum()) - 1.0
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    return X, y_base + noise

# %%
# Model Evaluation
# ---------------
#
# We'll evaluate the models using multiple metrics:
# - Pinball loss: Standard quantile regression loss
# - Percentage of positive residuals: Should match the target quantile
# - Sparsity: Percentage of zero coefficients
# - MAE and MSE: Additional error metrics


def evaluate_model(model, X_test, y_test, tau):
    """Evaluate model performance with multiple metrics."""
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    # Basic metrics
    loss = pinball_loss(y_test, y_pred, tau)
    pct_pos = np.mean(residuals > 0) * 100

    # Additional metrics
    sparsity = np.mean(np.abs(model.coef_) < 1e-10) * 100
    mae = np.mean(np.abs(residuals))
    mse = np.mean(residuals ** 2)

    return {
        'loss': loss,
        'pct_pos': pct_pos,
        'sparsity': sparsity,
        'mae': mae,
        'mse': mse
    }


def pinball_loss(y_true, y_pred, tau=0.5):
    """Compute Pinball (quantile) loss."""
    residuals = y_true - y_pred
    return np.mean(np.where(residuals >= 0,
                            tau * residuals,
                            (1 - tau) * -residuals))

# %%
# Performance Comparison
# --------------------
#
# Let's compare the performance across different problem sizes and noise
# distributions. This helps understand when the progressive smoothing
# approach is most beneficial.


# Test different problem sizes
problem_sizes = [
    (1000, 10),    # Small problem
    (5000, 100),   # Medium problem
    (10000, 1000)  # Large problem
]

alpha = 0.01

# Test different noise distributions
noise_types = ['exponential', 'student_t', 'mixture']

# Quantiles to test
quantiles = [0.1, 0.5, 0.9]

# Configure PDCD solver
pdcd_params = {
    'max_iter': 100,
    'tol': 1e-6,
    'fit_intercept': False,
    'warm_start': True,
    'p0': 50
}

# Store results
results = []

for n_samples, n_features in problem_sizes:
    for noise_type in noise_types:
        print(f"\n=== Testing {n_samples}x{n_features} with {noise_type} noise ===")

        # Generate data
        X, y = generate_data(n_samples, n_features, noise_type)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Choose SQR hyperparameters by problem size
        if n_features <= 10:
            # Small problem: very light smoothing, tiny working set
            sqr_params = dict(
                initial_delta=0.1,
                min_delta=1e-4,
                max_stages=4,
                quantile_error_threshold=0.01,
                solver_params={
                    'max_iter_start':  20,
                    'max_iter_step':   20,
                    'max_iter_cap':   100,
                    'large_problem_threshold': 1000,
                    'p0_frac_large':   0.1
                }
            )
        elif n_features <= 100:
            # Medium problem: moderate budget
            sqr_params = dict(
                initial_delta=0.1,
                min_delta=1e-4,
                max_stages=4,
                quantile_error_threshold=0.01,
                solver_params={
                    'max_iter_start':  50,
                    'max_iter_step':   50,
                    'max_iter_cap':   200,
                    'large_problem_threshold': 1000,
                    'p0_frac_large':   0.05
                }
            )
        else:
            # Large problem: lean budget
            sqr_params = dict(
                initial_delta=0.1,
                min_delta=1e-4,
                delta_tol=1e-4,
                max_stages=3,
                quantile_error_threshold=0.02,
                solver_params={
                    'base_tol': 1e-4,
                    'tol_delta_factor': 0.1,
                    'max_iter_start':  100,
                    'max_iter_step':   50,
                    'max_iter_cap':   1000,
                    'small_problem_threshold': 0,
                    'p0_frac_small':  0.02,
                    'large_problem_threshold': 1000,
                    'p0_frac_large':  0.1,
                }
            )

        for tau in quantiles:
            print(f"\n==== Quantile τ={tau} ====")

            # scikit-learn QuantileRegressor
            qr = QuantileRegressor(quantile=tau, alpha=alpha, solver="highs")
            t0 = time.time()
            qr.fit(X_train, y_train)
            qr_time = time.time() - t0
            qr_metrics = evaluate_model(qr, X_test, y_test, tau)

            # SmoothQuantileRegressor
            solver = PDCD_WS(**pdcd_params)
            sqr = SmoothQuantileRegressor(
                quantile=tau,
                alpha=alpha,
                alpha_schedule='geometric',
                initial_alpha=2 * alpha,
                verbose=False,
                smooth_solver=solver,
                **sqr_params
            )
            t1 = time.time()
            sqr.fit(X_train, y_train)
            sqr_time = time.time() - t1
            sqr_metrics = evaluate_model(sqr, X_test, y_test, tau)

            # Print results
            print(f"QR - Time: {qr_time:.2f}s, Loss: {qr_metrics['loss']:.4f}, "
                  f"% positive: {qr_metrics['pct_pos']:.1f}%, "
                  f"Sparsity: {qr_metrics['sparsity']:.1f}%")
            print(f"SQR - Time: {sqr_time:.2f}s, Loss: {sqr_metrics['loss']:.4f}, "
                  f"% positive: {sqr_metrics['pct_pos']:.1f}%, "
                  f"Sparsity: {sqr_metrics['sparsity']:.1f}%")

            # Store results
            results.append({
                'n_samples': n_samples,
                'n_features': n_features,
                'noise_type': noise_type,
                'tau': tau,
                'qr_time': qr_time,
                'sqr_time': sqr_time,
                'qr_loss': qr_metrics['loss'],
                'sqr_loss': sqr_metrics['loss'],
                'qr_pct_pos': qr_metrics['pct_pos'],
                'sqr_pct_pos': sqr_metrics['pct_pos'],
                'qr_sparsity': qr_metrics['sparsity'],
                'sqr_sparsity': sqr_metrics['sparsity'],
                'qr_mae': qr_metrics['mae'],
                'sqr_mae': sqr_metrics['mae'],
                'qr_mse': qr_metrics['mse'],
                'sqr_mse': sqr_metrics['mse']
            })

# Convert results to DataFrame
df = pd.DataFrame(results)

# Print summary statistics
print("\nOverall Performance Summary:")
summary = df.groupby(['n_samples', 'n_features', 'noise_type']).agg({
    'qr_time': 'mean',
    'sqr_time': 'mean',
    'qr_loss': 'mean',
    'sqr_loss': 'mean',
    'qr_pct_pos': 'mean',
    'sqr_pct_pos': 'mean',
    'qr_sparsity': 'mean',
    'sqr_sparsity': 'mean'
}).round(4)
print(summary)


# %%
# Visual Comparison
# ----------------
#
# Let's visualize the performance of both models on a representative case.
# We'll use a medium-sized problem with exponential noise to demonstrate
# the key differences.


# Generate data
n_samples, n_features = 5000, 100
X, y = generate_data(n_samples, n_features, 'exponential')
tau = 0.5
alpha = 0.01

solver = PDCD_WS(**pdcd_params)

# Fit models
qr = QuantileRegressor(quantile=tau, alpha=alpha, solver="highs")
qr.fit(X, y)
y_pred_qr = qr.predict(X)

sqr = SmoothQuantileRegressor(
    quantile=tau, alpha=alpha,
    alpha_schedule='geometric',
    initial_alpha=2 * alpha,     # milder continuation
    initial_delta=0.1,           # start closer to true loss
    min_delta=1e-4,             # stop sooner
    delta_tol=1e-4,             # allow earlier stage stopping
    max_stages=4,               # fewer smoothing stages
    quantile_error_threshold=0.01,  # coarser quantile error tolerance
    verbose=False,
    smooth_solver=solver,
).fit(X, y)
y_pred_sqr = sqr.predict(X)

# Compute residuals
residuals_qr = y - y_pred_qr
residuals_sqr = y - y_pred_sqr

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Sort data for better visualization
sort_idx = np.argsort(y)
y_sorted = y[sort_idx]
qr_pred = y_pred_qr[sort_idx]
sqr_pred = y_pred_sqr[sort_idx]

# Plot predictions
axes[0, 0].scatter(y_sorted, qr_pred, alpha=0.5, label='scikit-learn', s=10)
axes[0, 0].scatter(y_sorted, sqr_pred, alpha=0.5, label='SmoothQuantile', s=10)
axes[0, 0].plot([y_sorted.min(), y_sorted.max()],
                [y_sorted.min(), y_sorted.max()], 'k--', alpha=0.3)
axes[0, 0].set_xlabel('True values')
axes[0, 0].set_ylabel('Predicted values')
axes[0, 0].set_title(f'Predictions (τ={tau})')
axes[0, 0].legend()

# Plot residuals
axes[0, 1].hist(residuals_qr, bins=50, alpha=0.5, label='scikit-learn')
axes[0, 1].hist(residuals_sqr, bins=50, alpha=0.5, label='SmoothQuantile')
axes[0, 1].axvline(x=0, color='k', linestyle='--', alpha=0.3)
axes[0, 1].set_xlabel('Residuals')
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_title('Residual Distribution')
axes[0, 1].legend()

# Plot residuals vs predictions
axes[1, 0].scatter(y_pred_qr, residuals_qr, alpha=0.5, s=5, label='scikit-learn')
axes[1, 0].scatter(y_pred_sqr, residuals_sqr, alpha=0.5, s=5, label='SmoothQuantile')
axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[1, 0].set_xlabel('Predicted values')
axes[1, 0].set_ylabel('Residuals')
axes[1, 0].set_title('Residuals vs Predictions')
axes[1, 0].legend()

# Plot solution path for SmoothQuantileRegressor
if hasattr(sqr, 'stage_results_'):
    stages = sqr.stage_results_
    deltas = [s['delta'] for s in stages]
    errors = [s['quantile_error'] for s in stages]
    actual = [s['actual_quantile'] for s in stages]

    axes[1, 1].plot(deltas, errors, 'o-', label='Quantile Error')
    axes[1, 1].set_xlabel('Smoothing parameter (δ)')
    axes[1, 1].set_ylabel('Quantile Error')
    ax2 = axes[1, 1].twinx()
    ax2.plot(deltas, actual, 'r--', label='Actual Quantile')
    ax2.axhline(y=tau, color='g', linestyle=':', label=f'Target ({tau})')
    ax2.set_ylabel('Actual Quantile')

    # Combine legends
    lines1, labels1 = axes[1, 1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1, 1].legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    axes[1, 1].set_title('Convergence Path')
    axes[1, 1].set_xscale('log')
else:
    axes[1, 1].text(0.5, 0.5, 'Stage results not available',
                    horizontalalignment='center', verticalalignment='center',
                    transform=axes[1, 1].transAxes)

plt.tight_layout()
# %%
# Conclusion
# ---------
# NOTE: NOT FASTER FOR NOW THAN QUANTILE REGRESSOR. STILL NEED TO FIX THE PROBLEM
# The SmoothQuantileRegressor demonstrates significant speed improvements
# over scikit-learn's QuantileRegressor while maintaining similar accuracy.
# The progressive smoothing approach is particularly effective for:
#
# 1. Large datasets where direct optimization is challenging
# 2. Problems requiring multiple quantile levels
# 3. Cases where computational efficiency is crucial
#
# The key advantages are:
# - Faster convergence through progressive smoothing
# - Better handling of large-scale problems
# - Automatic adaptation to problem size
# - Maintained accuracy across different noise distributions
