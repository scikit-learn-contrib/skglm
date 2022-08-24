import numpy as np
from numba import njit


def gram_cd_solver(X, y, penalty, max_iter=20, tol=1e-4, verbose=False):
    """Run a Gram solver by reformulation the problem as below.

    Minimize::
        w.T @ Q @ w / (2*n_samples) - b.T @ w / n_samples + penalty(w)

    where::
        Q = X.T @ X
        b = X.T @ y
    """
    n_samples, n_features = X.shape
    XtX = X.T @ X / n_samples
    Xty = X.T @ y / n_samples
    all_features = np.arange(n_features)
    p_objs_out = []

    w = np.zeros(n_features)
    XtXw = np.zeros(n_features)
    # initial: grad = -Xty
    opt = penalty.subdiff_distance(w, -Xty, all_features)

    for t in range(max_iter):
        # check convergences
        stop_crit = np.max(opt)
        if verbose:
            p_obj = 0.5 * w @ XtXw - Xty @ w + penalty.value(w)
            print(
                f"Iteration {t+1}: {p_obj:.10f}, "
                f"stopping crit: {stop_crit:.2e}"
            )

        if stop_crit <= tol:
            if verbose:
                print(f"Stopping criterion max violation: {stop_crit:.2e}")
            break

        # inplace update of w, XtXw, opt
        _gram_cd_iter(XtX, Xty, w, XtXw, penalty, opt,
                      all_features, n_updates=n_features)

        p_obj = 0.5 * w @ XtXw - Xty @ w + penalty.value(w)
        p_objs_out.append(p_obj)
    return w, p_objs_out, stop_crit


@njit
def _gram_cd_iter(XtX, Xty, w, XtXw, penalty, opt, ws, n_updates):
    # inplace update of w, XtXw, opt
    for _ in range(n_updates):
        grad = XtXw - Xty
        opt[:] = penalty.subdiff_distance(w, grad, ws)
        j_max = np.argmax(opt)

        old_w_j = w[j_max]
        step = 1 / XtX[j_max, j_max]  # 1 / lipchitz_j
        w[j_max] = penalty.prox_1d(old_w_j - step * grad[j_max], step, j_max)

        # Gram matrix update
        if w[j_max] != old_w_j:
            XtXw += (w[j_max] - old_w_j) * XtX[:, j_max]
