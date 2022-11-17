import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

from celer import Lasso
from cd_solver.sklearn_api import Lasso as fb_Lasso
from skglm.utils import make_correlated_data

from prototype.pd_lasso import (fb_lasso, cp_lasso, forward_backward,
                                cd, _compute_obj)


EPS_FLOATING = 1e-10
regs = [1e-1, 1e-2, 1e-3]
n_samples, n_features = 300, 600
fig, axarr = plt.subplots(1, len(regs), sharey=False, figsize=[8., 3],
                          constrained_layout=True)

for reg, ax in zip(regs, axarr):
    A, b, _ = make_correlated_data(n_samples, n_features, random_state=24)

    n_samples, n_features = A.shape
    alpha_max = norm(A.T @ b, ord=np.inf)

    max_iter = 400
    alpha = reg * alpha_max

    print(f"========== reg = {reg} ================")
    # start = time.time()
    w_fb, p_objs_fb = fb_lasso(A, b, alpha, max_iter=max_iter)
    # end = time.time()
    # print("F&B time: ", end - start)

    # start = time.time()
    w_cp, p_objs_cp = cp_lasso(A, b, alpha, max_iter=max_iter)
    # end = time.time()
    # print("CB time: ", end - start)

    # start = time.time()
    # w, p_objs = forward_backward(A, b, alpha, max_iter=max_iter)
    # end = time.time()
    # print("forward backward time: ", end - start)

    # start = time.time()
    # w_cd, p_objs_cd = cd(A, b, alpha, max_iter=max_iter)
    # end = time.time()
    # print("cyclic CD time: ", end - start)

    # start = time.time()
    fq_estimator = fb_Lasso(alpha, smooth_formulation=False,
                            max_iter=max_iter)
    fq_estimator.fit(A, b)
    pb_obj_fb_lasso = fq_estimator.p_objs_
    # end = time.time()
    # print("cyclic CD time: ", end - start)

    # find optimal val
    lasso = Lasso(fit_intercept=False,
                  alpha=alpha / n_samples, tol=EPS_FLOATING).fit(A, b)
    w_start = lasso.coef_.flatten()
    p_star = _compute_obj(b, A, w_start, alpha) - EPS_FLOATING

    ax.semilogy(p_objs_fb - p_star, label="my Fercoq & Bianchi")
    ax.semilogy(pb_obj_fb_lasso - p_star, label="his Fercoq & Bianchi")
    ax.semilogy(p_objs_cp - p_star, label="Chambolle Pock")
    # ax.semilogy(p_objs - p_star, label="forward-backward")
    # ax.semilogy(p_objs_cd - p_star, label="cyclic CD")

    ax.legend()
    ax.set_xlabel("iteration")
    ax.set_title(f"reg={reg}")

fig.suptitle("Lasso on dataset with\n"
             f"n_samples={n_samples}, n_features={n_features}")
axarr[0].set_ylabel("primal suboptimality")
plt.show()
