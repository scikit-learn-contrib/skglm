# Authors: Can Pouliquen
#          Mathurin Massias
"""
=======================================================================
Regularization paths for the Graphical Lasso and its Adaptive variation
=======================================================================
This example demonstrates how non-convex penalties in the Adaptive Graphical Lasso
can achieve superior sparsity recovery compared to the standard L1 penalty.

The Adaptive Graphical Lasso uses iterative reweighting to approximate non-convex
penalties, following Candès et al. (2007). Non-convex penalties often produce
better sparsity patterns by more aggressively shrinking small coefficients while
preserving large ones.

We compare three approaches:
    - **L1**: Standard Graphical Lasso with L1 penalty
    - **Log**: Adaptive approach with logarithmic penalty
    - **L0.5**: Adaptive approach with L0.5 penalty

The plots show normalized mean square error (NMSE) for reconstruction accuracy
and F1 score for sparsity pattern recovery across different regularization levels.
"""

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from skglm.covariance import GraphicalLasso, AdaptiveGraphicalLasso
from skglm.penalties.separable import LogSumPenalty, L0_5
from skglm.utils.data import make_dummy_covariance_data

# %%
# Generate synthetic sparse precision matrix data
# ===============================================

p = 100
n = 1000
S, Theta_true, alpha_max = make_dummy_covariance_data(n, p)
alphas = alpha_max*np.geomspace(1, 1e-4, num=10)

# %%
# Setup models with different penalty functions
# ============================================

penalties = ["L1", "Log", "L0.5"]
n_reweights = 5  # Number of adaptive reweighting iterations
models_tol = 1e-4

models = [
    # Standard Graphical Lasso with L1 penalty
    GraphicalLasso(algo="primal", warm_start=True, tol=models_tol),

    # Adaptive Graphical Lasso with logarithmic penalty
    AdaptiveGraphicalLasso(warm_start=True,
                           penalty=LogSumPenalty(alpha=1.0, eps=1e-10),
                           n_reweights=n_reweights,
                           tol=models_tol),

    # Adaptive Graphical Lasso with L0.5 penalty
    AdaptiveGraphicalLasso(warm_start=True,
                           penalty=L0_5(alpha=1.0),
                           n_reweights=n_reweights,
                           tol=models_tol),
]

# %%
# Compute regularization paths
# ============================

nmse_results = {penalty: [] for penalty in penalties}
f1_results = {penalty: [] for penalty in penalties}


# Fit models across regularization path
for i, (penalty, model) in enumerate(zip(penalties, models)):
    print(f"Fitting {penalty} penalty across {len(alphas)} regularization values...")
    for alpha_idx, alpha in enumerate(alphas):
        print(
            f"  alpha {alpha_idx+1}/{len(alphas)}: "
            f"lambda/lambda_max = {alpha/alpha_max:.1e}",
            end="")

        model.alpha = alpha
        model.fit(S)

        Theta_est = model.precision_
        nmse = norm(Theta_est - Theta_true)**2 / norm(Theta_true)**2
        f1_val = f1_score(Theta_est.flatten() != 0., Theta_true.flatten() != 0.)

        nmse_results[penalty].append(nmse)
        f1_results[penalty].append(f1_val)

        print(f"NMSE: {nmse:.3f}, F1: {f1_val:.3f}")
    print(f"{penalty} penalty complete!\n")


# %%
# Plot results
# ============
fig, axarr = plt.subplots(2, 1, sharex=True, figsize=([6.11, 3.91]),
                          layout="constrained")
cmap = plt.get_cmap("tab10")
for i, penalty in enumerate(penalties):

    for j, ax in enumerate(axarr):

        if j == 0:
            metric = nmse_results
            best_idx = np.argmin(metric[penalty])
            ystop = np.min(metric[penalty])
        else:
            metric = f1_results
            best_idx = np.argmax(metric[penalty])
            ystop = np.max(metric[penalty])

        ax.semilogx(alphas/alpha_max,
                    metric[penalty],
                    color=cmap(i),
                    linewidth=2.,
                    label=penalty)

        ax.vlines(
            x=alphas[best_idx] / alphas[0],
            ymin=0,
            ymax=ystop,
            linestyle='--',
            color=cmap(i))
        line = ax.plot(
            [alphas[best_idx] / alphas[0]],
            0,
            clip_on=False,
            marker='X',
            color=cmap(i),
            markersize=12)

        ax.grid(which='both', alpha=0.9)

axarr[0].legend(fontsize=14)
axarr[0].set_title(f"{p=},{n=}", fontsize=18)
axarr[0].set_ylabel("NMSE", fontsize=18)
axarr[1].set_ylabel("F1 score", fontsize=18)
_ = axarr[1].set_xlabel(r"$\lambda / \lambda_\mathrm{{max}}$",  fontsize=18)
# %%
# Results summary
# ===============

print("Performance at optimal regularization:")
print("-" * 50)

for penalty in penalties:
    best_nmse = min(nmse_results[penalty])
    best_f1 = max(f1_results[penalty])
    print(f"{penalty:>4}: NMSE = {best_nmse:.3f}, F1 = {best_f1:.3f}")

# %% [markdown]
#
# **Metrics explanation:**
#
# * **NMSE (Normalized Mean Square Error)**: Measures reconstruction accuracy
#   of the precision matrix. Lower values = better reconstruction.
# * **F1 Score**: Measures sparsity pattern recovery (correctly identifying
#   which entries are zero/non-zero). Higher values = better sparsity.
#
# **Key finding**: Non-convex penalties achieve significantly
# better sparsity recovery (F1 score) while maintaining
# competitive reconstruction accuracy (NMSE).
