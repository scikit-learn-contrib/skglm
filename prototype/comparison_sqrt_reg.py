import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

from skglm.utils import make_correlated_data
from skglm.experimental.sqrt_lasso import SqrtLasso, _chambolle_pock_sqrt

from prototype.pd_sqrt_lasso import fb_sqrt_lasso, cp_sqrt_lasso, _compute_obj


EPS_FLOATING = 1e-10
regs = [1e-1, 1e-2, 1e-3]
n_samples, n_features = 1000, 500
fig, axarr = plt.subplots(1, len(regs), sharey=False, figsize=[8., 3],
                          constrained_layout=True)

for reg, ax in zip(regs, axarr):
    A, b, _ = make_correlated_data(n_samples, n_features, random_state=24)

    n_samples, n_features = A.shape
    alpha_max = norm(A.T @ b, ord=np.inf) / norm(b)
    sqrt_n = np.sqrt(n_samples)

    max_iter = 400
    alpha = reg * alpha_max

    print(f"========== reg = {reg} ================")
    w_fb, p_objs_fb = fb_sqrt_lasso(A, b, alpha, max_iter=max_iter)

    w_cp, p_objs_cp = cp_sqrt_lasso(A, b, alpha, max_iter=max_iter)

    # find optimal val
    lasso = SqrtLasso(alpha=alpha / sqrt_n, tol=EPS_FLOATING).fit(A, b)
    w_start = lasso.coef_.flatten()
    p_star = _compute_obj(b, A, w_start, alpha) - EPS_FLOATING

    ax.semilogy(p_objs_fb - p_star, label="my Fercoq & Bianchi")
    ax.semilogy(p_objs_cp - p_star, label="Chambolle Pock")

    ax.legend()
    ax.set_xlabel("iteration")
    ax.set_title(f"reg={reg}")

fig.suptitle("Sqrt Lasso on dataset with\n"
             f"n_samples={n_samples}, n_features={n_features}")
axarr[0].set_ylabel("primal suboptimality")
plt.show()
