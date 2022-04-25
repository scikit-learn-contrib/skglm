# Authors: Quentin Klopfenstein
#          Mathurin Massias
"""
=============================================
Timing comparison with scikit-learn for Lasso
=============================================
Compare time to solve large scale Lasso problems with scikit-learn.
"""


import time
import warnings
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from libsvmdata import fetch_libsvm

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso as Lasso_sklearn
from sklearn.linear_model import ElasticNet as Enet_sklearn

from skglm import Lasso, ElasticNet

warnings.filterwarnings('ignore', category=ConvergenceWarning)


def compute_obj(X, y, w, alpha, l1_ratio=1):
    loss = norm(y - X @ w) ** 2 / (2 * len(y))
    penalty = (alpha * l1_ratio * np.sum(np.abs(w))
               + 0.5 * alpha * (1 - l1_ratio) * norm(w) ** 2)
    return loss + penalty


X, y = fetch_libsvm("news20.binary"
                    )
alpha = np.max(np.abs(X.T @ y)) / len(y) / 50

dict_sklearn = {}
dict_sklearn["lasso"] = Lasso_sklearn(
    alpha=alpha, fit_intercept=False, tol=1e-12)

dict_sklearn["enet"] = Enet_sklearn(
    alpha=alpha, fit_intercept=False, tol=1e-12, l1_ratio=0.5)

dict_ours = {}
dict_ours["lasso"] = Lasso(
    alpha=alpha, fit_intercept=False, tol=1e-12)
dict_ours["enet"] = ElasticNet(
    alpha=alpha, fit_intercept=False, tol=1e-12, l1_ratio=0.5)

models = ["lasso", "enet"]

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
    w_star = dict_ours[model].fit(X, y).coef_
    pobj_star = compute_obj(X, y, w_star, alpha, l1_ratio)
    for n_iter_sklearn in np.unique(np.geomspace(1, 200, num=15).astype(int)):
        dict_sklearn[model].max_iter = n_iter_sklearn

        t_start = time.time()
        w_sklearn = dict_sklearn[model].fit(X, y).coef_
        time_dict["sklearn"].append(time.time() - t_start)
        pobj_dict["sklearn"].append(compute_obj(X, y, w_sklearn, alpha, l1_ratio))

    for n_iter_us in range(15):
        dict_ours[model].max_iter = n_iter_us
        t_start = time.time()
        w = dict_ours[model].fit(X, y).coef_
        time_dict["us"].append(time.time() - t_start)
        pobj_dict["us"].append(compute_obj(X, y, w, alpha, l1_ratio))

    ax.semilogy(
        time_dict["sklearn"], pobj_dict["sklearn"] - pobj_star, label='sklearn')
    ax.semilogy(
        time_dict["us"], pobj_dict["us"] - pobj_star, label='skglm')

    ax.set_ylim((1e-10, 1))
    ax.set_title(model)
    ax.legend()
    ax.set_ylabel("Objective suboptimality")

axarr[1].set_xlabel("Time (s)")
plt.show(block=False)
