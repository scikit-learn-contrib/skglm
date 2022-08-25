import numpy as np
from numba import njit
from skglm.utils import AndersonAcceleration


def gram_cd_solver(X, y, penalty, max_iter=20, w_init=None,
                   use_acc=True, tol=1e-4, verbose=False):
    """Run coordinate descent while keeping the gradients up-to-date with Gram updates.

    Minimize::
        1 / (2*n_samples) * norm(y - Xw)**2 + penalty(w)

    Which can be rewritten as::
        w.T @ Q @ w / (2*n_samples) - q.T @ w / n_samples + penalty(w)

    where::
        Q = X.T @ X (gram matrix)
        q = X.T @ y
    """
    n_samples, n_features = X.shape
    scaled_gram = X.T @ X / n_samples
    scaled_Xty = X.T @ y / n_samples
    scaled_y_norm2 = np.linalg.norm(y)**2 / (2*n_samples)
    all_features = np.arange(n_features)
    stop_crit = np.inf  # prevent ref before assign
    p_objs_out = []

    w = np.zeros(n_features) if w_init is None else w_init
    scaled_gram_w = np.zeros(n_features) if w_init is None else scaled_gram @ w_init
    opt = penalty.subdiff_distance(w, -scaled_Xty, all_features)  # initial: grad = -Xty
    if use_acc:
        accelerator = AndersonAcceleration(K=5)
        w_acc = np.zeros(n_features)
        scaled_gram_w_acc = np.zeros(n_features)

    for t in range(max_iter):
        # check convergences
        stop_crit = np.max(opt)
        if verbose:
            p_obj = (0.5 * w @ scaled_gram_w - scaled_Xty @ w +
                     scaled_y_norm2 + penalty.value(w))
            print(
                f"Iteration {t+1}: {p_obj:.10f}, "
                f"stopping crit: {stop_crit:.2e}"
            )

        if stop_crit <= tol:
            if verbose:
                print(f"Stopping criterion max violation: {stop_crit:.2e}")
            break

        # inplace update of w, XtXw
        opt = _gram_cd_epoch(scaled_gram, scaled_Xty, w, scaled_gram_w, penalty,
                             all_features)

        # perform anderson extrapolation
        if use_acc:
            w_acc, scaled_gram_w_acc, is_extrapolated = accelerator.extrapolate(
                w, scaled_gram_w)

            if is_extrapolated:
                p_obj_acc = (0.5 * w_acc @ scaled_gram_w_acc - scaled_Xty @ w_acc +
                             penalty.value(w_acc))
                p_obj = 0.5 * w @ scaled_gram_w - scaled_Xty @ w + penalty.value(w)
                if p_obj_acc < p_obj:
                    w[:] = w_acc
                    scaled_gram_w[:] = scaled_gram_w_acc

        p_obj = 0.5 * w @ scaled_gram_w - scaled_Xty @ w + penalty.value(w)
        p_objs_out.append(p_obj)
    return w, np.array(p_objs_out), stop_crit


@njit
def _gram_cd_epoch(scaled_gram, scaled_Xty, w, scaled_gram_w, penalty, ws):
    # inplace update of w, XtXw, opt
    # perform greedy cd updates
    for _ in range(len(w)):
        grad = scaled_gram_w - scaled_Xty
        opt = penalty.subdiff_distance(w, grad, ws)
        j_max = np.argmax(opt)

        old_w_j = w[j_max]
        step = 1 / scaled_gram[j_max, j_max]  # 1 / lipchitz_j
        w[j_max] = penalty.prox_1d(old_w_j - step * grad[j_max], step, j_max)

        # Gram matrix update
        if w[j_max] != old_w_j:
            scaled_gram_w += (w[j_max] - old_w_j) * scaled_gram[:, j_max]
    return opt
