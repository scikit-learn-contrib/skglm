import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.utils import check_random_state

from skglm.estimators import GraphicalLasso
from skglm.estimators import AdaptiveGraphicalLasso

# Data
p = 100
n = 1000
rng = check_random_state(0)
Theta_true = make_sparse_spd_matrix(
    p,
    alpha=0.9,
    random_state=rng)

Theta_true += 0.1*np.eye(p)
Sigma_true = np.linalg.pinv(Theta_true, hermitian=True)
X = rng.multivariate_normal(
    mean=np.zeros(p),
    cov=Sigma_true,
    size=n,
)

S = np.cov(X, bias=True, rowvar=False)
S_cpy = np.copy(S)
np.fill_diagonal(S_cpy, 0.)
alpha_max = np.max(np.abs(S_cpy))

alphas = alpha_max*np.geomspace(1, 1e-4, num=10)


penalties = [
    "L1",
    "R-L1 (log)",
    "R-L1 (L0.5)",
    "R-L1 (MCP)",
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
    [6.11, 3.91]), layout="constrained")
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
# plt.savefig(f"./non_convex_p{p}_n{n}.pdf")
plt.show(block=False)
