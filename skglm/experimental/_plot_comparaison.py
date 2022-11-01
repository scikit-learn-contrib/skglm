import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from skglm.utils import make_correlated_data, compiled_clone
from skglm.experimental.sqrt_lasso import SqrtLasso, _chambolle_pock_sqrt

from skglm.penalties import L1
from skglm.experimental.sqrt_lasso import SqrtQuadratic
from skglm.experimental.fercoq_bianchi import fercoq_bianchi

import time


def find_p_star(*args):
    global X, y, alpha
    EPS_FLOATING = 1e-10

    def sqrt_lasso_obj(w):
        return norm(X @ w - y) / np.sqrt(len(y)) + alpha * norm(w, ord=1)

    p_objs = [sqrt_lasso_obj(coef)
              for coef in args]

    # run sqrt lasso
    clf = SqrtLasso(alpha=alpha, tol=1e-10)
    clf.fit(X, y)

    p_objs.append(sqrt_lasso_obj(clf.coef_.flatten()))

    return min(p_objs) - EPS_FLOATING


fig, axarr = plt.subplots(1, 2, sharey=False, figsize=[8., 3],
                          constrained_layout=True)
# for normalize, ax in zip([False, True], axarr):
for normalize, ax in zip([False, True], axarr):
    X, y, _ = make_correlated_data(n_samples=100, n_features=100, random_state=24)
    if normalize:
        X /= norm(X, axis=0)

    n_samples, n_features = X.shape
    alpha_max = norm(X.T @ y, ord=np.inf) / (norm(y) * np.sqrt(n_samples))

    alpha = alpha_max / 10
    max_iter = 1000

    # cache numba
    datafit = compiled_clone(SqrtQuadratic())
    penalty = compiled_clone(L1(alpha))
    # fercoq_bianchi(X, y, datafit, penalty, max_iter=2)

    print(f"========== {normalize} ================")
    start = time.time()
    w_cd, objs_cd, _ = fercoq_bianchi(
        X, y, datafit, penalty, max_iter=max_iter, tol=1e-6)
    end = time.time()
    # print("F&B time: ", end - start)

    start = time.time()
    w, _, objs = _chambolle_pock_sqrt(
        X, y, alpha, max_iter=max_iter, obj_freq=10)
    end = time.time()
    # print("CB time: ", end - start)

    # plt.close('all')
    p_star = find_p_star(w, w_cd)
    ax.semilogy(objs - p_star, label="CP")
    ax.semilogy(objs_cd - p_star, label="FB")
    ax.legend()
    ax.set_xlabel("iteration")
    ax.set_title(f"Normalize={normalize}")
axarr[0].set_ylabel("primal suboptimality")
plt.show()
