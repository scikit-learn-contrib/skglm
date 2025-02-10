import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.covariance import GraphicalLasso as skGraphicalLasso

from skglm.estimators import GraphicalLasso, AdaptiveGraphicalLasso
from skglm.utils.data import generate_GraphicalLasso_data

# Data
p = 20
n = 100
S, Theta_true, alpha_max = generate_GraphicalLasso_data(n, p)

alphas = alpha_max*np.geomspace(1, 1e-3, num=30)


penalties = [
    "L1",
    "R-L1",
]

models_tol = 1e-4
models = [
    GraphicalLasso(algo="mazumder",
                   warm_start=True, tol=models_tol),
    AdaptiveGraphicalLasso(warm_start=True, n_reweights=10, tol=models_tol),

]

my_glasso_nmses = {penalty: [] for penalty in penalties}
my_glasso_f1_scores = {penalty: [] for penalty in penalties}

sk_glasso_nmses = []
sk_glasso_f1_scores = []


for i, (penalty, model) in enumerate(zip(penalties, models)):
    print(penalty)
    for alpha_idx, alpha in enumerate(alphas):
        print(f"======= alpha {alpha_idx+1}/{len(alphas)} =======")
        model.alpha = alpha
        model.fit(S)
        Theta = model.precision_

        my_nmse = norm(Theta - Theta_true)**2 / norm(Theta_true)**2

        my_f1_score = f1_score(Theta.flatten() != 0.,
                               Theta_true.flatten() != 0.)
        print(f"NMSE: {my_nmse:.3f}")
        print(f"F1  : {my_f1_score:.3f}")

        my_glasso_nmses[penalty].append(my_nmse)
        my_glasso_f1_scores[penalty].append(my_f1_score)


plt.close('all')
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(
    [12.6,  4.63]), layout="constrained")
cmap = plt.get_cmap("tab10")
for i, penalty in enumerate(penalties):

    ax[0].semilogx(alphas/alpha_max,
                   my_glasso_nmses[penalty],
                   color=cmap(i),
                   linewidth=2.,
                   label=penalty)
    min_nmse = np.argmin(my_glasso_nmses[penalty])
    ax[0].vlines(
        x=alphas[min_nmse] / alphas[0],
        ymin=0,
        ymax=np.min(my_glasso_nmses[penalty]),
        linestyle='--',
        color=cmap(i))
    line0 = ax[0].plot(
        [alphas[min_nmse] / alphas[0]],
        0,
        clip_on=False,
        marker='X',
        color=cmap(i),
        markersize=12)

    ax[1].semilogx(alphas/alpha_max,
                   my_glasso_f1_scores[penalty],
                   linewidth=2.,
                   color=cmap(i))
    max_f1 = np.argmax(my_glasso_f1_scores[penalty])
    ax[1].vlines(
        x=alphas[max_f1] / alphas[0],
        ymin=0,
        ymax=np.max(my_glasso_f1_scores[penalty]),
        linestyle='--',
        color=cmap(i))
    line1 = ax[1].plot(
        [alphas[max_f1] / alphas[0]],
        0,
        clip_on=False,
        marker='X',
        markersize=12,
        color=cmap(i))


ax[0].set_title(f"{p=},{n=}", fontsize=18)
ax[0].set_ylabel("NMSE", fontsize=18)
ax[1].set_ylabel("F1 score", fontsize=18)
ax[1].set_xlabel(f"$\lambda / \lambda_\mathrm{{max}}$",  fontsize=18)

ax[0].legend(fontsize=14)
ax[0].grid(which='both', alpha=0.9)
ax[1].grid(which='both', alpha=0.9)
# plt.show(block=False)
plt.show()
