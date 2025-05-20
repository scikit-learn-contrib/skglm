"""
=======================================
Quantile Huber Loss Visualization
=======================================

This example demonstrates the smoothed approximation of the pinball loss
(Quantile Huber) used in quantile regression. It illustrates how the
smoothing parameter affects the loss function shape and optimization
performance for different quantile levels.
"""

# %%
# Understanding the Quantile Huber Loss
# ------------------------------------
#
# The standard pinball loss used in quantile regression is not differentiable
# at zero, which causes issues for gradient-based optimization methods.
# The Quantile Huber loss provides a smoothed approximation by replacing the
# non-differentiable point with a quadratic region.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
import time

from skglm import GeneralizedLinearEstimator
from skglm.penalties import L1
from skglm.solvers import FISTA
from skglm.experimental.quantile_huber import QuantileHuber

# %%
# First, let's visualize the Quantile Huber loss for median regression (τ=0.5)
# with different smoothing parameters.

fig1, ax1 = plt.subplots(figsize=(10, 5))

# Plot several smoothing levels
deltas = [1.0, 0.5, 0.2, 0.1, 0.01]
tau = 0.5
r = np.linspace(-3, 3, 1000)

# Calculate the non-smooth pinball loss for comparison
pinball_loss = np.where(r >= 0, tau * r, (tau - 1) * r)
ax1.plot(r, pinball_loss, 'k--', label='Pinball (non-smooth)')

# Plot losses for different deltas
for delta in deltas:
    loss_fn = QuantileHuber(delta=delta, quantile=tau)
    loss = np.array([loss_fn._loss_and_grad_scalar(ri)[0] for ri in r])
    ax1.plot(r, loss, '-', label=f'Quantile Huber (delta={delta})')

ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.4)
ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.4)
ax1.set_xlabel('Residual (r)')
ax1.set_ylabel('Loss')
ax1.set_title('Quantile Huber Loss (τ=0.5) with Different Smoothing Parameters')
ax1.legend()
ax1.grid(True, alpha=0.3)

# %%
# As we can see, smaller values of delta make the approximation closer to the
# original non-smooth pinball loss. Now, let's compare different quantile
# levels with the same smoothing parameter.

fig2, ax2 = plt.subplots(figsize=(10, 5))

# Plot for different quantile levels
taus = [0.1, 0.25, 0.5, 0.75, 0.9]
delta = 0.2

for tau in taus:
    loss_fn = QuantileHuber(delta=delta, quantile=tau)
    loss = np.array([loss_fn._loss_and_grad_scalar(ri)[0] for ri in r])
    ax2.plot(r, loss, '-', label=f'τ={tau}')

ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.4)
ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.4)
ax2.set_xlabel('Residual (r)')
ax2.set_ylabel('Loss')
ax2.set_title(f'Quantile Huber Loss for Different Quantile Levels (delta={delta})')
ax2.legend()
ax2.grid(True, alpha=0.3)

# %%
# The gradient of the smoothed loss
# ---------------------------------
#
# The primary advantage of the Quantile Huber loss is its continuous gradient.
# Let's visualize both the loss and its gradient for specific quantile levels.


def plot_quantile_huber(tau=0.5, delta=0.5, residual_range=(-2, 2), num_points=1000):
    """
    Plot the quantile Huber loss function and its gradient.

    This utility function generates plots of the quantile Huber loss and its
    gradient for visualization and documentation purposes. It's not part of the
    core implementation but is useful for understanding the behavior of the loss.

    Parameters
    ----------
    tau : float, default=0.5
        Quantile level between 0 and 1.

    delta : float, default=0.5
        Smoothing parameter controlling the width of the quadratic region.

    residual_range : tuple (min, max), default=(-2, 2)
        Range of residual values to plot.

    num_points : int, default=1000
        Number of points to plot.

    Returns
    -------
    fig : matplotlib figure
        Figure containing the plots.

    Example
    -------
    >>> from skglm.experimental.quantile_huber import plot_quantile_huber
    >>> fig = plot_quantile_huber(tau=0.8, delta=0.3)
    >>> fig.savefig('quantile_huber_tau_0.8.png')
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        raise ImportError("Matplotlib is required for plotting.")

    loss_fn = QuantileHuber(delta=delta, quantile=tau)
    r = np.linspace(residual_range[0], residual_range[1], num_points)

    # Calculate loss and gradient for each residual value
    loss = np.zeros_like(r)
    grad = np.zeros_like(r)

    for i, ri in enumerate(r):
        loss_val, grad_val = loss_fn._loss_and_grad_scalar(ri)
        loss[i] = loss_val
        grad[i] = grad_val

    # For comparison, calculate the non-smooth pinball loss
    pinball_loss = np.where(r >= 0, tau * r, (tau - 1) * r)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot loss functions
    ax1.plot(r, loss, 'b-', label=f'Quantile Huber (δ={delta})')
    ax1.plot(r, pinball_loss, 'r--', label='Pinball (non-smooth)')

    # Add vertical lines at the transition points
    ax1.axvline(x=delta, color='gray', linestyle=':', alpha=0.7)
    ax1.axvline(x=-delta, color='gray', linestyle=':', alpha=0.7)
    ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

    # Add horizontal line at y=0
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    # Add shaded regions to highlight the quadratic zone
    ax1.axvspan(-delta, delta, alpha=0.1, color='blue')

    ax1.set_xlabel('Residual (r)')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Quantile Huber Loss (τ={tau})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot gradients
    ax2.plot(r, grad, 'b-', label=f'Quantile Huber Gradient (δ={delta})')

    # Add the non-smooth gradient for comparison
    pinball_grad = np.where(r > 0, tau, tau - 1)
    ax2.plot(r, pinball_grad, 'r--', label='Pinball Gradient (non-smooth)')

    # Add vertical lines at the transition points
    ax2.axvline(x=delta, color='gray', linestyle=':', alpha=0.7)
    ax2.axvline(x=-delta, color='gray', linestyle=':', alpha=0.7)
    ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

    # Add horizontal line at y=0
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    # Add shaded regions to highlight the quadratic zone
    ax2.axvspan(-delta, delta, alpha=0.1, color='blue')

    ax2.set_xlabel('Residual (r)')
    ax2.set_ylabel('Gradient')
    ax2.set_title(f'Gradient of Quantile Huber Loss (τ={tau})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# For the 75th percentile with delta=0.3
fig3 = plot_quantile_huber(tau=0.75, delta=0.3)
plt.suptitle('Quantile Huber Loss and Gradient for τ=0.75, delta=0.3', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle

# %%
# For the 50th percentile (Median) with delta=0.3
fig4 = plot_quantile_huber(tau=0.5, delta=0.3)
plt.suptitle('Quantile Huber Loss and Gradient for τ=0.5, delta=0.3', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle

# %%
# For the 25th percentile with delta=0.3
fig4 = plot_quantile_huber(tau=0.25, delta=0.3)
plt.suptitle('Quantile Huber Loss and Gradient for τ=0.25, delta=0.3', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle

# %%
# Performance impact of smoothing parameter
# -----------------------------------------
#
# The smoothing parameter delta strongly affects optimization performance.
# Let's examine how it impacts convergence speed and solution quality.

# Generate synthetic data
X, y = make_regression(n_samples=500, n_features=10, noise=0.1, random_state=42)
X = StandardScaler().fit_transform(X)

# Set up the experiment
delta_values = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02]
tau = 0.75
alpha = 0.1

# Collect results
times = []
objectives = []

# Define pinball loss function for evaluation


def pinball_loss(y_true, y_pred, tau=0.5):
    """Compute the pinball loss for quantile regression."""
    residuals = y_true - y_pred
    return np.mean(np.where(residuals >= 0,
                            tau * residuals,
                            (1 - tau) * -residuals))


# Run for each delta
for delta in delta_values:
    datafit = QuantileHuber(delta=delta, quantile=tau)
    solver = FISTA(max_iter=1000, tol=1e-6)

    start_time = time.time()
    est = GeneralizedLinearEstimator(
        datafit=datafit, penalty=L1(alpha=alpha), solver=solver
    ).fit(X, y)

    elapsed = time.time() - start_time
    times.append(elapsed)

    # Compute pinball loss of the solution (not the smoothed loss)
    pinball = pinball_loss(y, X @ est.coef_, tau=tau)
    objectives.append(pinball)

# %%
# The results show the trade-off between optimization speed and solution quality.
# Let's plot the results to visualize this relationship.

fig5, ax5 = plt.subplots(figsize=(10, 5))

ax5.plot(delta_values, times, 'o-')
ax5.set_xscale('log')
ax5.set_xlabel('Delta (delta)')
ax5.set_ylabel('Time (seconds)')
ax5.set_title('Computation Time vs Smoothing Parameter')
ax5.grid(True, alpha=0.3)

fig6, ax7 = plt.subplots(figsize=(10, 5))
ax7.plot(delta_values, objectives, 'o-')
ax7.set_xscale('log')
ax7.set_xlabel('Delta (delta)')
ax7.set_ylabel('Pinball Loss')
ax7.set_title(f'Final Pinball Loss (τ={tau}) vs Smoothing Parameter')
ax7.grid(True, alpha=0.3)

# %%
# This example illustrates the key trade-off when choosing the smoothing parameter:
#
# - Larger values of delta make the problem easier to optimize (faster convergence,
#   fewer iterations), but may yield less accurate results for the original
#   quantile regression objective.
# - Smaller values of delta give more accurate results, but may require more iterations
#   to converge and take longer to compute.
#
# In practice, a progressive smoothing approach (as used in SmoothQuantileRegressor)
# can be beneficial, starting with a large delta and gradually reducing it to
# approach the original non-smooth problem.
#
# The optimal choice of delta depends on your specific application and the balance
# between computational efficiency and solution accuracy you require.
