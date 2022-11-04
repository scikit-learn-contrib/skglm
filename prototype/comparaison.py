import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso
from skglm.utils import make_correlated_data

from prototype.pd_lasso import fb_lasso, cp_lasso, _compute_obj


EPS_FLOATING = 1e-10
reg = 1e-1
n_samples, n_features = 100, 100
fig, axarr = plt.subplots(1, 2, sharey=False, figsize=[8., 3],
                          constrained_layout=True)

for normalize, ax in zip([False, True], axarr):
    A, b, _ = make_correlated_data(n_samples, n_features, random_state=24)
    if normalize:
        A /= norm(b, axis=0)

    n_samples, n_features = A.shape
    alpha_max = norm(A.T @ b, ord=np.inf)

    alpha = reg * alpha_max

    print(f"========== {normalize} ================")
    # start = time.time()
    w_fb, p_objs_fb = fb_lasso(A, b, alpha, max_iter=10000)
    # end = time.time()
    # print("F&B time: ", end - start)

    # start = time.time()
    w_cp, p_objs_cp = cp_lasso(A, b, alpha, max_iter=10000)
    # end = time.time()
    # print("CB time: ", end - start)

    lasso = Lasso(fit_intercept=False,
                  alpha=alpha / n_samples).fit(A, b)
    w_start = lasso.coef_.flatten()
    p_star = _compute_obj(b, A, w_start, alpha)

    ax.semilogy(p_objs_cp - p_star, label="CP")
    ax.semilogy(p_objs_fb - p_star, label="FB")

    ax.legend()
    ax.set_xlabel("iteration")
    ax.set_title(f"Normalize={normalize}")

fig.suptitle(f"n_samples={n_samples}, n_features={n_features}, reg={reg}")
axarr[0].set_ylabel("primal suboptimality")
plt.show()
