import numpy as np
from numba import njit


def gram_solver(X, y, penalty, max_iter=20, max_epoch=1000, p0=10, tol=1e-4,
                verbose=False):
    """Run a Gram solver by reformulation the problem as below.

    Minimize::
        w.T @ Q @ w / (2*n_samples) - b.T @ w / n_samples + penalty(w)

    where::
        Q = X.T @ X
        b = X.T @ y
    """
    n_features = X.shape[1]
    XtX = X.T @ X
    Xty = X.T @ y
    all_features = np.arange(n_features)
    p_objs_out = []

    w = np.zeros(n_features)
    XtXw = np.zeros(n_features)

    for t in range(max_iter):
        # compute scores
        grad = _construct_grad(y, XtXw, Xty, all_features)
        opt = penalty.subdiff_distance(w, grad, all_features)

        # check convergences
        stop_crit = np.max(opt)
        if verbose:
            p_obj = _quadratic_value(X, w, XtXw, Xty) + penalty.value(w)
            print(
                f"Iteration {t+1}: {p_obj:.10f}, "
                f"stopping crit: {stop_crit:.2e}"
            )

        if stop_crit <= tol:
            if verbose:
                print(f"Stopping criterion max violation: {stop_crit:.2e}")
            break

        # build ws
        gsupp_size = penalty.generalized_support(w).sum()
        ws_size = max(min(p0, n_features),
                      min(n_features, 2 * gsupp_size))
        # similar to np.argsort()[-ws_size:] but without sorting
        ws = np.argpartition(opt, -ws_size)[-ws_size:]
        tol_in = 0.3 * stop_crit

        for epoch in range(max_epoch):
            # inplace update of w, XtXw
            _gram_cd_epoch(y, XtX, Xty, w, XtXw, penalty, ws)

            if epoch % 10 == 0:
                grad = _construct_grad(y, XtXw, Xty, ws)
                opt_in = penalty.subdiff_distance(w, grad, ws)

                stop_crit_in = np.max(opt_in)
                if max(verbose-1, 0):
                    p_obj = _quadratic_value(X, w, XtXw, Xty) + penalty.value(w)
                    print(
                        f"Epoch {epoch+1}: {p_obj:.10f}, "
                        f"stopping crit in: {stop_crit_in:.2e}"
                    )

                if stop_crit_in <= tol_in:
                    if max(verbose-1, 0):
                        print("Early exit")
                    break

        p_obj = _quadratic_value(X, w, XtXw, Xty) + penalty.value(w)
        p_objs_out.append(p_obj)
    return w, p_objs_out, stop_crit


@njit
def _gram_cd_epoch(y, XtX, Xty, w, XtXw, penalty, ws):
    # inplace update of w, XtXw
    for j in ws:
        # skip for X[:, j] == 0
        if XtX[j, j] == 0:
            continue

        old_w_j = w[j]
        grad_j = (XtXw[j] - Xty[j]) / len(y)
        step = 1 / XtX[j, j]  # 1 / lipchitz_j

        w[j] = penalty.prox_1d(old_w_j - step * grad_j, step, j)

        # Gram matrix update
        if w[j] != old_w_j:
            XtXw += (w[j] - old_w_j) * XtX[:, j]


@njit
def _construct_grad(y, XtXw, Xty, ws):
    n_samples = len(y)
    grad = np.zeros(len(ws))
    for idx, j in enumerate(ws):
        grad[idx] = (XtXw[j] - Xty[j]) / n_samples
    return grad


@njit
def _quadratic_value(X, w, XtXw, Xty):
    n_samples = X.shape[0]
    return w @ XtXw / (2*n_samples) - Xty @ w / n_samples
