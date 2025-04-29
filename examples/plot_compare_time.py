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
alpha = np.max(np.abs(X.T @ y)) / len(y) / 10

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

# Global warmup to ensure all libraries are JIT-compiled before any benchmarking
print("Performing global warmup to trigger JIT compilation...")
for model_name in models:
    if model_name == "lasso":
        warmup_sklearn = Lasso_sklearn(alpha=alpha, fit_intercept=False, tol=1e-12)
        warmup_skglm = Lasso(alpha=alpha, fit_intercept=False, tol=1e-12)
    else:  # enet
        warmup_sklearn = Enet_sklearn(
            alpha=alpha, fit_intercept=False, tol=1e-12, l1_ratio=0.5)
        warmup_skglm = ElasticNet(
            alpha=alpha, fit_intercept=False, tol=1e-12, l1_ratio=0.5)

    print(f"Warming up {model_name} models...")
    # Use the full dataset for warmup to ensure all code paths are compiled
    _ = warmup_sklearn.fit(X, y)
    _ = warmup_skglm.fit(X, y)

# Now start the actual benchmarking
fig, axarr = plt.subplots(2, 1, constrained_layout=True)

for ax, model, l1_ratio in zip(axarr, models, [1, 0.5]):
    pobj_dict = {}
    pobj_dict["sklearn"] = list()
    pobj_dict["us"] = list()

    time_dict = {}
    time_dict["sklearn"] = list()
    time_dict["us"] = list()

    # Perform warmup runs with a small subset to trigger compilation
    _ = dict_sklearn[model].fit(X[:10], y[:10])
    _ = dict_ours[model].fit(X[:10], y[:10])

    # Find optimal solution for reference
    dict_ours[model].max_iter = 10_000
    w_star = dict_ours[model].fit(X, y).coef_
    pobj_star = compute_obj(X, y, w_star, alpha, l1_ratio)

    # Reset models with fresh instances after using them for reference solution
    if model == "lasso":
        dict_sklearn[model] = Lasso_sklearn(
            alpha=alpha, fit_intercept=False, tol=1e-12)
        dict_ours[model] = Lasso(
            alpha=alpha, fit_intercept=False, tol=1e-12)
    else:  # enet
        dict_sklearn[model] = Enet_sklearn(
            alpha=alpha, fit_intercept=False, tol=1e-12, l1_ratio=0.5)
        dict_ours[model] = ElasticNet(
            alpha=alpha, fit_intercept=False, tol=1e-12, l1_ratio=0.5)

    # # --------------------
    # # DEBUG: measure compile vs. solver iteration cost
    # debug_model = dict_ours[model]

    # # 1) compile + first iteration
    # debug_model.max_iter = 1
    # t0 = time.time()
    # _ = debug_model.fit(X, y)
    # debug_total = time.time() - t0
    # print(f"[DEBUG] {model}: compile+1 iter = {debug_total:.3f}s")

    # # 2) only solver iteration (post-compile)
    # debug_model.warm_start = True
    # debug_model.max_iter = 1
    # t0 = time.time()
    # _ = debug_model.fit(X, y)
    # debug_iter = time.time() - t0
    # print(f"[DEBUG] {model}: first iter post-compile = {debug_iter:.3f}s")

    # # 3) iteration cost with fixpoint update
    # debug_model.warm_start = False
    # debug_model.ws_strategy = "fixpoint"
    # debug_model.max_iter = 1
    # t0 = time.time()
    # _ = debug_model.fit(X, y)
    # debug_fixpoint = time.time() - t0
    # print(f"[DEBUG] {model}: first iter fixpoint = {debug_fixpoint:.3f}s")
    # # --------------------

    # warm up JIT so that no compile goes into your timing measurements
    print("warming up skglm on a little subset to pay the compile cost up‐front…")
    X_small, y_small = X[:10], y[:10]
    _ = Lasso(alpha=alpha, fit_intercept=False, tol=1e-12).fit(X_small, y_small)
    _ = ElasticNet(alpha=alpha, fit_intercept=False, tol=1e-12,
                   l1_ratio=0.5).fit(X_small, y_small)

    print("Warmup complete!")

    for n_iter_sklearn in np.unique(np.geomspace(1, 50, num=15).astype(int)):
        dict_sklearn[model].max_iter = n_iter_sklearn
        print(f"  sklearn iterations: {n_iter_sklearn}")

        t_start = time.time()
        w_sklearn = dict_sklearn[model].fit(X, y).coef_

        time_dict["sklearn"].append(time.time() - t_start)
        pobj_dict["sklearn"].append(compute_obj(X, y, w_sklearn, alpha, l1_ratio))

    for n_iter_us in np.unique(np.geomspace(1, 50, num=15).astype(int)):
        dict_ours[model].max_iter = n_iter_us
        print(f"  skglm iterations: {n_iter_us}")

        t_start = time.time()
        w = dict_ours[model].fit(X, y).coef_

        time_dict["us"].append(time.time() - t_start)
        pobj_dict["us"].append(compute_obj(X, y, w, alpha, l1_ratio))

    ax.semilogy(
        time_dict["sklearn"], pobj_dict["sklearn"] - pobj_star, marker='o', label='sklearn')
    ax.semilogy(
        time_dict["us"], pobj_dict["us"] - pobj_star, marker='o', label='skglm')

    ax.set_ylim((1e-10, 1))
    ax.set_title(model)
    ax.legend()
    ax.set_ylabel("Objective suboptimality")

axarr[1].set_xlabel("Time (s)")
plt.show()
