# Authors: Can Pouliquen
#          Mathurin Massias
"""
=======================================================================
Regularization paths for the Graphical Lasso and its Adaptive variation
=======================================================================
Highlight the importance of using non-convex regularization for improved performance,
solved using the reweighting strategy.
"""

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from skglm.covariance import GraphicalLasso, AdaptiveGraphicalLasso
from skglm.utils.data import make_dummy_covariance_data


p = 100
n = 1000
S, Theta_true, alpha_max = make_dummy_covariance_data(n, p)
alphas = alpha_max*np.geomspace(1, 1e-4, num=10)

penalties = [
    "L1",
    "Log",
    "L0.5",
    "MCP",
]
n_reweights = 5
models_tol = 1e-4
models = [
    GraphicalLasso(algo="primal",
                   warm_start=True,
                   tol=models_tol),
    AdaptiveGraphicalLasso(warm_start=True,
                           strategy="log",
                           n_reweights=n_reweights,
                           tol=models_tol),
    AdaptiveGraphicalLasso(warm_start=True,
                           strategy="sqrt",
                           n_reweights=n_reweights,
                           tol=models_tol),
    AdaptiveGraphicalLasso(warm_start=True,
                           strategy="mcp",
                           n_reweights=n_reweights,
                           tol=models_tol),
]

my_glasso_nmses = {penalty: [] for penalty in penalties}
my_glasso_f1_scores = {penalty: [] for penalty in penalties}

sk_glasso_nmses = []
sk_glasso_f1_scores = []


for i, (penalty, model) in enumerate(zip(penalties, models)):
    for alpha_idx, alpha in enumerate(alphas):
        print(f"======= {penalty} penalty, alpha {alpha_idx+1}/{len(alphas)} =======")
        model.alpha = alpha
        model.fit(S)
        Theta = model.precision_

        my_nmse = norm(Theta - Theta_true)**2 / norm(Theta_true)**2

        my_f1_score = f1_score(Theta.flatten() != 0.,
                               Theta_true.flatten() != 0.)

        my_glasso_nmses[penalty].append(my_nmse)
        my_glasso_f1_scores[penalty].append(my_f1_score)


plt.close('all')
fig, axarr = plt.subplots(2, 1, sharex=True, figsize=([6.11, 3.91]),
                          layout="constrained")
cmap = plt.get_cmap("tab10")
for i, penalty in enumerate(penalties):

    for j, ax in enumerate(axarr):

        if j == 0:
            metric = my_glasso_nmses
            best_idx = np.argmin(metric[penalty])
            ystop = np.min(metric[penalty])
        else:
            metric = my_glasso_f1_scores
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
axarr[1].set_xlabel(r"$\lambda / \lambda_\mathrm{{max}}$",  fontsize=18)

plt.show(block=False)
