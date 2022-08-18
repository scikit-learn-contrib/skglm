import numpy as np
from scipy.sparse import issparse
from numba import njit
from skglm.py_numba_blitz.utils import (
    compute_primal_obj, compute_dual_obj,
    compute_remaining_features,
    update_XTtheta, update_XTtheta_s, update_phi_XTphi,
    update_theta_exp_yXw, weighted_dot_sparse,
    norm2_sparse, xj_dot_sparse, squared_weighted_norm_sparse,
    update_X_delta_w, LOGREG_LIPSCHITZ_CONST
)
from skglm.utils import ST


MAX_BACKTRACK_ITER = 20
MAX_PROX_NEWTON_CD_ITR = 20
MIN_PROX_NEWTON_CD_ITR = 2
PROX_NEWTON_EPSILON_RATIO = 10.0
EPSILON_GAP = 0.3


def py_blitz(alpha, X, y, p0=100, max_iter=20, max_epochs=100,
             tol=1e-9, verbose=False, sort_ws=False):
    r"""Solve Logistic Regression.

    Objective:

        \sum_{i=1}^n log(1 + e^{-y_i (Xw)_i}) + \alpha \sum_{j=1}^p |w_j|
    """
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    Xw = np.zeros(n_samples)
    exp_yXw = np.zeros(n_samples)  # np.exp(-y * Xw)

    # dual vars
    theta = np.zeros(n_samples)
    theta_scale = 1.
    phi = np.zeros(n_samples)
    XTtheta = np.zeros(n_features)
    XTphi = np.zeros(n_features)

    remaining_features = np.arange(n_features)
    norm2_X_cols = np.zeros(n_features)
    is_sparse = issparse(X)

    # init vars
    if is_sparse:
        X_bundles = (X.data, X.indptr, X.indices)
    update_theta_exp_yXw(y, Xw, theta, exp_yXw)
    for j in range(n_features):
        if is_sparse:
            norm2_X_cols[j] = norm2_sparse(*X_bundles, j)
        else:
            norm2_X_cols[j] = np.linalg.norm(X[:, j], ord=2)

    # main loop
    for t in range(max_iter):
        if is_sparse:
            update_XTtheta_s(*X_bundles, theta, XTtheta, remaining_features)
        else:
            update_XTtheta(X, theta, XTtheta, remaining_features)

        update_phi_XTphi(theta_scale * theta, theta_scale * XTtheta, phi,
                         XTphi, alpha, remaining_features)

        p_obj = compute_primal_obj(exp_yXw, w, alpha)
        d_obj = compute_dual_obj(y, phi)
        gap = p_obj - d_obj
        prev_p_obj = p_obj

        threshold = np.sqrt(2 * gap / LOGREG_LIPSCHITZ_CONST)
        remaining_features = compute_remaining_features(remaining_features, XTphi,
                                                        w, norm2_X_cols, alpha,
                                                        threshold)

        # The output of sorting algo might differ for an array
        # which some of its elements are equal (e.g [0, 1, 5, 0, 0, 8, 0])
        # We and Blitz uses sorting algos to build ws and hence may get different
        # ws which result into different logs throughout the epochs/iterations
        # This ensure that we have the same ws (for a comparison purposes with Blitz)
        if sort_ws:
            remaining_features.sort()

        ws_size = min(
            max(2*np.sum(w != 0), p0),
            len(remaining_features)
        )

        # prox newton vars
        prox_grad_diff = 0.
        prox_grads = np.zeros(ws_size)
        for idx, j in enumerate(remaining_features[:ws_size]):
            if is_sparse:
                prox_grads[idx] = xj_dot_sparse(*X_bundles, j, theta)
            else:
                prox_grads[idx] = X[:, j] @ theta

        for epoch in range(max_epochs):
            max_cd_iter = MAX_PROX_NEWTON_CD_ITR if epoch else MIN_PROX_NEWTON_CD_ITR
            prev_p_obj_in = p_obj

            if is_sparse:
                (
                    theta_scale,
                    prox_grad_diff
                ) = _prox_newton_iteration_s(*X_bundles, y, w, Xw, exp_yXw,
                                             theta, prox_grads, alpha,
                                             ws=remaining_features[:ws_size],
                                             max_cd_iter=max_cd_iter,
                                             prox_grad_diff=prox_grad_diff)
            else:
                (
                    theta_scale,
                    prox_grad_diff
                ) = _prox_newton_iteration(X, y, w, Xw, exp_yXw,
                                           theta, prox_grads, alpha,
                                           ws=remaining_features[:ws_size],
                                           max_cd_iter=max_cd_iter,
                                           prox_grad_diff=prox_grad_diff)

            p_obj = compute_primal_obj(exp_yXw, w, alpha)
            d_obj_in = compute_dual_obj(y, theta_scale * theta)
            gap_in = p_obj - d_obj_in

            if verbose:
                print(
                    f"|—— Epoch: {epoch+1} "
                    f"Objective: {p_obj} "
                    f"Duality gap: {p_obj - d_obj} "
                )

            if gap_in < EPSILON_GAP * (p_obj - d_obj):
                break
            elif gap_in / np.abs(d_obj_in) < tol:
                break
            elif p_obj >= prev_p_obj_in:
                break

        p_obj = compute_primal_obj(exp_yXw, w, alpha)
        gap = p_obj - d_obj

        if verbose:
            print(
                f"Iter {t+1}: "
                f"Objective: {p_obj} "
                f"Duality gap: {gap} "
                f"Feature left {len(remaining_features[:ws_size])} "
                # f"ws: {remaining_features[:ws_size]} "
            )

        if gap / np.abs(d_obj) < tol:
            break
        elif p_obj >= prev_p_obj:
            break

    return w


@njit
def _prox_newton_iteration(X, y, w, Xw, exp_yXw, theta, prox_grads,
                           alpha, ws, max_cd_iter, prox_grad_diff):
    # inplace update of w, Xw, exp_yXw, theta, prox_grads

    hessian = -theta * (y + theta)  # \nabla^2 datafit(u)
    lipschitz = np.zeros(len(ws))  # diag of X.T hessian X

    delta_w = np.zeros(len(ws))  # descent direction
    X_delta_w = np.zeros(len(y))

    for idx, j in enumerate(ws):
        lipschitz[idx] = hessian @ X[:, j] ** 2

    # find descent direction
    for cd_iter in range(max_cd_iter):
        sum_sq_hess_diff = 0.
        for idx, j in enumerate(ws):
            # skip zero cols
            if lipschitz[idx] == 0:
                continue

            old_w_j = w[j] + delta_w[idx]
            grad = prox_grads[idx] + X[:, j] @ (hessian * X_delta_w)
            step = 1 / lipschitz[idx]
            new_w_j = ST(old_w_j - step * grad, alpha * step)

            # updates
            diff = new_w_j - old_w_j
            if diff == 0:
                continue

            delta_w[idx] = new_w_j - w[j]
            X_delta_w += diff * X[:, j]
            sum_sq_hess_diff += (diff * lipschitz[idx]) ** 2

        if (sum_sq_hess_diff < PROX_NEWTON_EPSILON_RATIO*prox_grad_diff
                and cd_iter+1 >= MIN_PROX_NEWTON_CD_ITR):
            break

    # backtracking line search
    actual_t, prev_t = 1., 0.
    for backtrack_iter in range(MAX_BACKTRACK_ITER):
        diff_objectives = 0.
        diff_t = actual_t - prev_t

        for idx, j in enumerate(ws):
            w[j] += diff_t * delta_w[idx]

            # diff penalty term
            if w[j] < 0:
                diff_objectives -= alpha * delta_w[idx]
            elif w[j] > 0:
                diff_objectives += alpha * delta_w[idx]
            else:
                diff_objectives -= alpha * abs(delta_w[idx])

        for i in range(len(y)):
            Xw[i] += diff_t * X_delta_w[i]
        update_theta_exp_yXw(y, Xw, theta, exp_yXw)

        # diff datafit term
        diff_objectives += X_delta_w @ theta

        if diff_objectives < 0:
            break
        else:
            prev_t = actual_t
            actual_t /= 2

    if actual_t != 1.:
        X_delta_w[:] = actual_t * X_delta_w

    # cache grads next epoch
    for idx, j in enumerate(ws):
        new_prox_grad = X[:, j] @ theta
        approximate_grad = prox_grads[idx] + X[:, j] @ (hessian * X_delta_w)
        prox_grads[idx] = new_prox_grad

        prox_grad_diff += (new_prox_grad - approximate_grad) ** 2

    # compute theta scale
    theta_scale = 1.
    max_XTtheta_ws = np.linalg.norm(prox_grads, ord=np.inf)
    if max_XTtheta_ws > alpha:
        theta_scale = alpha / max_XTtheta_ws

    return theta_scale, prox_grad_diff


@njit
def _prox_newton_iteration_s(X_data, X_indptr, X_indices, y, w, Xw, exp_yXw,
                             theta, prox_grads, alpha, ws, max_cd_iter,
                             prox_grad_diff):
    # inplace update of w, Xw, exp_yXw, theta, prox_grads

    hessian = -theta * (y + theta)  # \nabla^2 datafit(u)
    lipschitz = np.zeros(len(ws))  # diag of X.T hessian X

    delta_w = np.zeros(len(ws))  # descent direction
    X_delta_w = np.zeros(len(y))

    for idx, j in enumerate(ws):
        lipschitz[idx] = squared_weighted_norm_sparse(X_data, X_indptr,
                                                      X_indices, hessian, j)

    # find descent direction
    for cd_iter in range(max_cd_iter):
        sum_sq_hess_diff = 0.
        for idx, j in enumerate(ws):
            # skip zero cols
            if lipschitz[idx] == 0:
                continue

            old_w_j = w[j] + delta_w[idx]
            grad = prox_grads[idx] + weighted_dot_sparse(X_data, X_indptr, X_indices,
                                                         X_delta_w, hessian, j)
            step = 1 / lipschitz[idx]
            new_w_j = ST(old_w_j - step * grad, alpha * step)

            # updates
            diff = new_w_j - old_w_j
            if diff == 0:
                continue

            delta_w[idx] = new_w_j - w[j]
            # for i in range(X_indptr[j], X_indptr[j+1]):
            #     X_delta_w[X_indices[i]] += diff * X_data[i]
            # equivalent to: X_delta_w += diff * X[:, j]
            update_X_delta_w(X_data, X_indptr, X_indices, X_delta_w, diff, j)
            sum_sq_hess_diff += (diff * lipschitz[idx]) ** 2

        if (sum_sq_hess_diff < PROX_NEWTON_EPSILON_RATIO*prox_grad_diff
                and cd_iter+1 >= MIN_PROX_NEWTON_CD_ITR):
            break

    # backtracking line search
    actual_t, prev_t = 1., 0.
    for backtrack_iter in range(MAX_BACKTRACK_ITER):
        diff_objectives = 0.
        diff_t = actual_t - prev_t

        for idx, j in enumerate(ws):
            w[j] += diff_t * delta_w[idx]

            # diff penalty term
            if w[j] < 0:
                diff_objectives -= alpha * delta_w[idx]
            elif w[j] > 0:
                diff_objectives += alpha * delta_w[idx]
            else:
                diff_objectives -= alpha * abs(delta_w[idx])

        for i in range(len(y)):
            Xw[i] += diff_t * X_delta_w[i]
        update_theta_exp_yXw(y, Xw, theta, exp_yXw)

        # diff datafit term
        diff_objectives += X_delta_w @ theta

        if diff_objectives < 0:
            break
        else:
            prev_t = actual_t
            actual_t /= 2

    if actual_t != 1.:
        X_delta_w[:] = actual_t * X_delta_w

    # cache grads next epoch
    for idx, j in enumerate(ws):
        new_prox_grad = xj_dot_sparse(X_data, X_indptr, X_indices, j, theta)
        approximate_grad = prox_grads[idx] + weighted_dot_sparse(X_data, X_indptr,
                                                                 X_indices, X_delta_w,
                                                                 hessian, j)
        prox_grads[idx] = new_prox_grad
        prox_grad_diff += (new_prox_grad - approximate_grad) ** 2

    # compute theta scale
    theta_scale = 1.
    max_XTtheta_ws = np.linalg.norm(prox_grads, ord=np.inf)
    if max_XTtheta_ws > alpha:
        theta_scale = alpha / max_XTtheta_ws

    return theta_scale, prox_grad_diff
