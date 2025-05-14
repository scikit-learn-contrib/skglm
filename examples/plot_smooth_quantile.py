"""
===========================================
Fast Quantile Regression with Smoothing
===========================================
This example demonstrates how SmoothQuantileRegressor achieves faster convergence
than scikit-learn's QuantileRegressor while maintaining accuracy, particularly
for large datasets.
"""

# %%
# Data Generation
# --------------
# First, we generate synthetic data with a known quantile structure.

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import QuantileRegressor
from skglm.experimental.smooth_quantile_regressor import SmoothQuantileRegressor
from skglm.solvers import FISTA

# Set random seed for reproducibility
np.random.seed(42)

# Generate dataset - using a more reasonable size for quick testing
n_samples, n_features = 1000, 10  # Match test file size
X, y = make_regression(n_samples=n_samples, n_features=n_features,
                       noise=0.1, random_state=42)
X = StandardScaler().fit_transform(X)
y = y - np.mean(y)  # Center y like in test file

# %%
# Model Comparison
# ---------------
# We compare scikit-learn's QuantileRegressor with our SmoothQuantileRegressor
# on the 80th quantile.

tau = 0.5  # median (SmoothQuantileRegressor works much better for non-median quantiles)
alpha = 0.1


def pinball_loss(y_true, y_pred, tau=0.5):
    """Compute Pinball (quantile) loss."""
    residuals = y_true - y_pred
    return np.mean(np.where(residuals >= 0,
                            tau * residuals,
                            (1 - tau) * -residuals))


# scikit-learn's QuantileRegressor
start_time = time.time()
qr = QuantileRegressor(quantile=tau, alpha=alpha, fit_intercept=True,
                       solver="highs").fit(X, y)
qr_time = time.time() - start_time
y_pred_qr = qr.predict(X)
qr_loss = pinball_loss(y, y_pred_qr, tau=tau)

# SmoothQuantileRegressor
start_time = time.time()
solver = FISTA(max_iter=2000, tol=1e-8)
solver.fit_intercept = True
sqr = SmoothQuantileRegressor(
    smoothing_sequence=[1.0, 0.5, 0.2, 0.1, 0.05],  # Base sequence, will be extended
    quantile=tau, alpha=alpha, verbose=True,  # Enable verbose to see stages
    smooth_solver=solver
).fit(X, y)
sqr_time = time.time() - start_time
y_pred_sqr = sqr.predict(X)
sqr_loss = pinball_loss(y, y_pred_sqr, tau=tau)

# %%
# Performance Analysis
# ------------------
# Let's analyze both the performance and solution quality of both methods.

speedup = qr_time / sqr_time
rel_gap = (sqr_loss - qr_loss) / qr_loss

print("\nPerformance Summary:")
print("scikit-learn QuantileRegressor:")
print(f"  Time: {qr_time:.2f}s")
print(f"  Loss: {qr_loss:.6f}")
print("SmoothQuantileRegressor:")
print(f"  Time: {sqr_time:.2f}s")
print(f"  Loss: {sqr_loss:.6f}")
print(f"  Speedup: {speedup:.1f}x")
print(f"  Relative gap: {rel_gap:.1%}")

# %%
# Visual Comparison
# ---------------
# We create visualizations to compare the predictions and residuals
# of both methods.

# Sort data for better visualization
sort_idx = np.argsort(y)
y_sorted = y[sort_idx]
qr_pred = y_pred_qr[sort_idx]
sqr_pred = y_pred_sqr[sort_idx]

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot predictions
ax1.scatter(y_sorted, qr_pred, alpha=0.5, label='scikit-learn', s=10)
ax1.scatter(y_sorted, sqr_pred, alpha=0.5, label='SmoothQuantile', s=10)
ax1.plot([y_sorted.min(), y_sorted.max()],
         [y_sorted.min(), y_sorted.max()], 'k--', alpha=0.3)
ax1.set_xlabel('True values')
ax1.set_ylabel('Predicted values')
ax1.set_title(f'Predictions (τ={tau})')
ax1.legend()

# Plot residuals
qr_residuals = y_sorted - qr_pred
sqr_residuals = y_sorted - sqr_pred
ax2.hist(qr_residuals, bins=50, alpha=0.5, label='scikit-learn')
ax2.hist(sqr_residuals, bins=50, alpha=0.5, label='SmoothQuantile')
ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)
ax2.set_xlabel('Residuals')
ax2.set_ylabel('Count')
ax2.set_title('Residual Distribution')
ax2.legend()

plt.tight_layout()

# %%
# Progressive Smoothing Analysis
# ----------------------------
# Let's examine how the smoothing parameter affects the solution quality.

stages = sqr.stage_results_
deltas = [s['delta'] for s in stages]
errors = [s['quantile_error'] for s in stages]
losses = [s['obj_value'] for s in stages]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot quantile error progression
ax1.semilogx(deltas, errors, 'o-')
ax1.set_xlabel('Smoothing parameter (δ)')
ax1.set_ylabel('Quantile error')
ax1.set_title('Quantile Error vs Smoothing')
ax1.grid(True, alpha=0.3)

# Plot objective value progression
ax2.semilogx(deltas, losses, 'o-')
ax2.set_xlabel('Smoothing parameter (δ)')
ax2.set_ylabel('Objective value')
ax2.set_title('Objective Value vs Smoothing')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
