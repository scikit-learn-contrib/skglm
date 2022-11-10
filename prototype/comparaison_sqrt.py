import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

from skglm.utils import make_correlated_data
from skglm.experimental.sqrt_lasso import SqrtLasso, _chambolle_pock_sqrt

from prototype.pd_sqrt_lasso import fb_sqrt_lasso, cp_sqrt_lasso, _compute_obj


EPS_FLOATING = 1e-10
reg = 1e-1
n_samples, n_features = 1000, 100
fig, axarr = plt.subplots(1, 2, sharey=False, figsize=[8., 3],
                          constrained_layout=True)

for normalize, ax in zip([False, True], axarr):
    A, b, _ = make_correlated_data(n_samples, n_features, random_state=24)
    if normalize:
        A /= norm(A, axis=0)

    n_samples, n_features = A.shape
    alpha_max = norm(A.T @ b, ord=np.inf) / norm(b)
    sqrt_n = np.sqrt(n_samples)

    max_iter = 400
    alpha = reg * alpha_max

    print(f"========== {normalize} ================")
    w_fb, p_objs_fb = fb_sqrt_lasso(A, b, alpha, max_iter=max_iter)

    w_cp, p_objs_cp = cp_sqrt_lasso(A, b, alpha, max_iter=max_iter)

    w_norm, _, p_objs_norm = _chambolle_pock_sqrt(
        A, b, alpha / sqrt_n, max_iter=max_iter)

    # find optimal val
    lasso = SqrtLasso(alpha=alpha / sqrt_n, tol=EPS_FLOATING).fit(A, b)
    w_start = lasso.coef_.flatten()
    p_star = _compute_obj(b, A, w_start, alpha) - EPS_FLOATING

    ax.semilogy(p_objs_fb - p_star, label="Fercoq & Bianchi")
    ax.semilogy(p_objs_cp - p_star, label="Chambolle Pock")
    ax.semilogy(sqrt_n * p_objs_norm - p_star, label="normalized Chambolle Pock")

    ax.legend()
    ax.set_xlabel("iteration")
    ax.set_title(f"Normalize={normalize}")

fig.suptitle(f"n_samples={n_samples}, n_features={n_features}, reg={reg}")
axarr[0].set_ylabel("primal suboptimality")
plt.show()
