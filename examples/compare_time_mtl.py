# Authors: Quentin Bertrand
#          Pierre-Antoine Bannier
"""
=============================================
Timing comparison with scikit-learn for MultiTask Lasso
=============================================
Compare time to solve medium scale MultiTask Lasso problems with scikit-learn.
"""


import time
import warnings
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from libsvmdata import fetch_libsvm

from skglm.utils import make_correlated_data
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import MultiTaskLasso as MultiTaskLasso_sklearn

from skglm import MultiTaskLasso

warnings.filterwarnings('ignore', category=ConvergenceWarning)


def compute_obj(X, Y, W, alpha, l1_ratio=1):
    loss = norm(Y - X @ W) ** 2 / (2 * Y.shape[0])
    penalty = (alpha * l1_ratio * np.sum(norm(W, axis=1))
               + 0.5 * alpha * (1 - l1_ratio) * (W ** 2).sum())
    return loss + penalty

n_features = 1000
n_samples = 1000
n_tasks = 10
X, Y, W_true = make_correlated_data(
    n_samples=n_samples, n_features=n_features, n_tasks=n_tasks,
    random_state=0)


alpha = np.max(norm(X.T @ Y, ord=2, axis=1)) / Y.shape[0] / 10

dict_sklearn = {}
dict_sklearn["lasso"] = MultiTaskLasso_sklearn(
    alpha=alpha, fit_intercept=False, tol=1e-12)

dict_ours = {}
dict_ours["lasso"] = MultiTaskLasso(
    alpha=alpha, fit_intercept=False, tol=1e-12, verbose=1)

models = ["lasso"]

fig, axarr = plt.subplots(2, 1, constrained_layout=True)

for ax, model, l1_ratio in zip(axarr, models, [1, 0.5]):
    pobj_dict = {}
    pobj_dict["sklearn"] = list()
    pobj_dict["us"] = list()

    time_dict = {}
    time_dict["sklearn"] = list()
    time_dict["us"] = list()

    # Remove compilation time
    dict_ours[model].max_iter = 10_000
    w_star = dict_ours[model].fit(X, Y).coef_.T
    pobj_star = compute_obj(X, Y, w_star, alpha, l1_ratio)
    for n_iter_sklearn in np.unique(np.geomspace(1, 50, num=15).astype(int)):
        dict_sklearn[model].max_iter = n_iter_sklearn

        t_start = time.time()
        w_sklearn = dict_sklearn[model].fit(X, Y).coef_.T
        time_dict["sklearn"].append(time.time() - t_start)
        pobj_dict["sklearn"].append(compute_obj(
            X, Y, w_sklearn, alpha, l1_ratio))

    for n_iter_us in range(1, 30):
        dict_ours[model].max_iter = n_iter_us
        t_start = time.time()
        w = dict_ours[model].fit(X, Y).coef_.T
        time_dict["us"].append(time.time() - t_start)
        pobj_dict["us"].append(compute_obj(X, Y, w, alpha, l1_ratio))

    ax.semilogy(
        time_dict["sklearn"], pobj_dict["sklearn"] - pobj_star, label='sklearn')
    ax.semilogy(
        time_dict["us"], pobj_dict["us"] - pobj_star, label='skglm')

    # ax.set_ylim((1e-10, 1))
    ax.set_title(model)
    ax.legend()
    ax.set_ylabel("Objective suboptimality")

axarr[1].set_xlabel("Time (s)")
plt.show(block=False)
