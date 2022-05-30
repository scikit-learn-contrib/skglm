import numpy as np
from numba import njit


def grp_bcd_solver(X, y, datafit, penalty,
                   max_iter=1000, tol=1e-7, verbose=False):
    n_samples, n_features = X.shape
    n_groups = len(penalty.grp_partition) - 1

    # init
    w = np.zeros(n_features)
    Xw = np.zeros(n_samples)  # X @ w in general
    datafit.initialize(X, y)
    all_groups = np.arange(n_groups)

    for k in range(max_iter):
        prev_p_obj = datafit.value(y, w, Xw) + penalty.value(w)
        _bcd_epoch(X, y, w, Xw, datafit, penalty, all_groups)
        current_obj = datafit.value(y, w, Xw) + penalty.value(w)

        if verbose:
            print(f"Iteration {k}: {current_obj}")

        if np.abs(current_obj - prev_p_obj) < tol:  # naive stopping criterion
            print("Early exit")
            break

    return w


@njit
def _bcd_epoch(X, y, w, Xw, datafit, penalty, ws):
    grp_partition, grp_indices = penalty.grp_partition, penalty.grp_indices

    for g in ws:
        grp_g_indices = grp_indices[grp_partition[g]: grp_partition[g+1]]
        old_w_g = w[grp_g_indices].copy()

        inv_lipschitz_g = 1 / datafit.lipschitz[g]
        grad_g = datafit.gradient_g(X, y, w, Xw, g)

        w[grp_g_indices] = penalty.prox_1group(
            old_w_g - inv_lipschitz_g * grad_g,
            inv_lipschitz_g, g
        )

        # update Xw without copying w_g
        for i, g_i in enumerate(grp_g_indices):
            if old_w_g[i] != w[g_i]:
                Xw += (w[g_i] - old_w_g[i]) * X[:, g_i]
    return
