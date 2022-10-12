
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from skglm.utils import make_correlated_data
from skglm.experimental.sqrt_lasso import SqrtLasso, _chambolle_pock_sqrt, _fercoq_bianchi_sqrt


fig, axarr = plt.subplots(1, 2, sharey=True, figsize=[8., 3],
                          constrained_layout=True)
# for normalize, ax in zip([False, True], axarr):
for normalize, ax in zip([True, False], axarr):
    X, y, _ = make_correlated_data(n_samples=1000, n_features=500, random_state=24)
    if normalize:
        X /= norm(X, axis=0) ** 2

    n_samples, n_features = X.shape
    alpha_max = norm(X.T @ y, ord=np.inf) / (norm(y) * np.sqrt(n_samples))

    alpha = alpha_max / 10

    max_iter = 1000
    obj_freq = 10
    verbose = False
    w_cd, _, objs_cd = _fercoq_bianchi_sqrt(
        X, y, alpha, max_iter=max_iter, obj_freq=obj_freq, verbose=verbose)
    w, _, objs = _chambolle_pock_sqrt(
        X, y, alpha, max_iter=max_iter, obj_freq=obj_freq, verbose=verbose)

    # no convergence issue if n_features < n_samples, can use ProxNewton
    # clf = SqrtLasso(alpha=alpha / np.sqrt(n_samples), verbose=2, tol=1e-10)
    clf = SqrtLasso(alpha=alpha, verbose=2, tol=1e-10)
    clf.fit(X, y)

    # consider that our solver has converged
    w_star = clf.coef_
    p_star = norm(X @ w_star - y) / np.sqrt(n_samples) + alpha * norm(w_star, ord=1)

    # p_star = min(np.min(objs), np.min(objs_cd))
    # plt.close('all')
    ax.semilogy(np.arange(1, max_iter+1, obj_freq), objs - p_star, label="CP")
    ax.semilogy(np.arange(1, max_iter+1, obj_freq),
                objs_cd - p_star, label="Fercoq Bianchi ")
    ax.legend()
    ax.set_xlabel("iteration")
    ax.set_title(f"Normalize={normalize}")
axarr[0].set_ylabel("primal suboptimality")
plt.show(block=False)


# plt.close("all")
# plt.semilogy(np.arange(1, max_iter+1, obj_freq), np.array(objs) - p_star)
# plt.xlabel("CP iteration")
# plt.ylabel("$F(x) - F(x^*)$")
# plt.show(block=False)
