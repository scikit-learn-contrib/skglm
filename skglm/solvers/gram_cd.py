import numpy as np
from numba import njit
from skglm.utils import AndersonAcceleration


def gram_cd_solver(X, y, penalty, max_iter=20, use_acc=True, tol=1e-4, verbose=False):
    """Run a Gram solver by reformulation the problem as below.

    Minimize::
        w.T @ Q @ w / (2*n_samples) - q.T @ w / n_samples + penalty(w)

    where::
        Q = X.T @ X
        q = X.T @ y
    """
    n_samples, n_features = X.shape
    XtX = X.T @ X / n_samples
    Xty = X.T @ y / n_samples
    all_features = np.arange(n_features)
    stop_crit = np.inf
    p_objs_out = []

    w = np.zeros(n_features)
    XtXw = np.zeros(n_features)
    opt = penalty.subdiff_distance(w, -Xty, all_features)  # initial: grad = -Xty
    if use_acc:
        accelerator = AndersonAcceleration(K=5)
        w_acc = np.zeros(n_features)
        XtXw_acc = np.zeros(n_features)

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

        # perform anderson extrapolation
        if use_acc:
            w_acc, XtXw_acc, is_extrapolated = accelerator.extrapolate(w, XtXw)

            if is_extrapolated:
                p_obj_acc = 0.5 * w_acc @ XtXw_acc - Xty @ w_acc + penalty.value(w_acc)
                p_obj = 0.5 * w @ XtXw - Xty @ w + penalty.value(w)
                if p_obj_acc < p_obj:
                    w[:] = w_acc
                    XtXw[:] = XtXw_acc

        p_obj = 0.5 * w @ XtXw - Xty @ w + penalty.value(w)
        p_objs_out.append(p_obj)
    return w, np.array(p_objs_out), stop_crit


@njit
def _gram_cd_iter(XtX, Xty, w, XtXw, penalty, opt, ws, n_updates):
    # inplace update of w, XtXw, opt
    # perform greedy cd updates
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
