import numpy as np

from py_numba_blitz.utils import(compute_primal_obj, compute_dual_obj,
                                 compute_remaining_features,
                                 update_XTtheta, update_phi_XTphi,
                                 update_theta_exp_yXw, LOGREG_LIPSCHITZ_CONST)

from skglm.utils import ST


MAX_BACKTRACK_ITER = 20
MAX_PROX_NEWTON_CD_ITR = 20
MIN_PROX_NEWTON_CD_ITR = 2
PROX_NEWTON_EPSILON_RATIO = 10.0
EPSILON_GAP = 0.3


def py_blitz(alpha, X, y, p0=100, max_iter=20, max_epochs=100, tol=1e-9, verbose=False):
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

    # init vars
    update_theta_exp_yXw(y, Xw, theta, exp_yXw)
    for j in range(n_features):
        norm2_X_cols[j] = np.linalg.norm(X[:, j], ord=2)

    for t in range(max_iter):
        update_XTtheta(X, theta, XTtheta, remaining_features)
        update_phi_XTphi(theta_scale * theta, theta_scale * XTtheta, phi,
                         XTphi, alpha, remaining_features)

        p_obj = compute_primal_obj(exp_yXw, w, alpha)
        d_obj = compute_dual_obj(y, phi)
        gap = p_obj - d_obj
        prev_p_obj = p_obj

        threshold = np.sqrt(2 * gap / LOGREG_LIPSCHITZ_CONST)
        remaining_features = compute_remaining_features(remaining_features, XTphi,
                                                        w, norm2_X_cols, alpha, threshold)

        ws_size = min(
            max(2*np.sum(w != 0), p0),
            len(remaining_features)
        )

        if verbose:
            print(
                f"B | Iter {t}: "
                f"Primal: {p_obj} "
                f"Dual: {d_obj} "
                f"Gap: {gap} "
                f"Feature left {ws_size} "
                f"ws: {remaining_features[:ws_size]} "
            )

        # prox newton vars
        prox_tol = 0.
        prox_grads = np.zeros(ws_size)
        for idx, j in enumerate(remaining_features[:ws_size]):
            prox_grads[idx] = X[:, j] @ theta

        print(f"Iter {t}:========")

        for epoch in range(max_epochs):
            max_cd_iter = MAX_PROX_NEWTON_CD_ITR if epoch else MIN_PROX_NEWTON_CD_ITR
            prev_p_obj_in = p_obj

            theta_scale, prox_tol = _prox_newton_iteration(X, y, w, Xw, exp_yXw, theta,
                                                           prox_grads, alpha,
                                                           ws=remaining_features[:ws_size],
                                                           max_cd_iter=max_cd_iter,
                                                           prox_tol=prox_tol)

            p_obj = compute_primal_obj(exp_yXw, w, alpha)
            d_obj_in = compute_dual_obj(y, theta_scale * theta)
            gap_in = p_obj - d_obj_in

            if gap_in < EPSILON_GAP * (p_obj - d_obj):
                print("Exit 1")
                break
            elif gap_in / np.abs(d_obj_in) < tol:
                print("Exit 2")
                break
            elif p_obj >= prev_p_obj_in:
                print("Exit 3")
                break

        p_obj = compute_primal_obj(exp_yXw, w, alpha)
        gap = p_obj - d_obj

        if verbose:
            print(
                f"Iter {t}: "
                f"Primal: {p_obj} "
                f"Dual: {d_obj} "
                f"Gap: {gap} "
                f"Feature left {ws_size} "
                f"ws: {remaining_features[:ws_size]} "
            )

        if gap / np.abs(d_obj) < tol:
            break
        elif p_obj >= prev_p_obj:
            break

    return w, Xw


def _prox_newton_iteration(X, y, w, Xw, exp_yXw, theta, prox_grads, alpha, ws, max_cd_iter, prox_tol):
    hessian = - theta * (y + theta)  # \nabla^2 datafit(u)
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

            # print(f"cd {cd_iter}: w[{j}]={new_w_j}")

            # updates
            diff = new_w_j - old_w_j
            if diff == 0:
                continue

            delta_w[idx] = new_w_j - w[j]
            X_delta_w += diff * X[:, j]
            sum_sq_hess_diff += (diff * lipschitz[idx]) ** 2

        if (sum_sq_hess_diff < prox_tol and cd_iter+1 > MIN_PROX_NEWTON_CD_ITR):
            break

    # backtracking line search
    actual_t, prev_t = 1., 0.
    diff_objectives = 0.
    for backtrack_iter in range(MAX_BACKTRACK_ITER):
        diff_t = actual_t - prev_t

        for idx, j in enumerate(ws):
            w[j] += diff_t * delta_w[idx]

            # print(f"w[{j}]={w[j]}")

            # diff penalty term
            if w[j] == 0:
                diff_objectives += alpha * np.abs(delta_w[idx])
            else:
                diff_objectives += np.sign(w[j]) * alpha * delta_w[idx]

        Xw[:] += diff_t * X_delta_w
        update_theta_exp_yXw(y, Xw, theta, exp_yXw)

        # diff datafit term
        diff_objectives += X_delta_w @ theta

        if diff_objectives < 0:
            break
        else:
            actual_t /= 2

    # print("backtrack", actual_t)
    if actual_t != 1.:
        X_delta_w[:] = actual_t * X_delta_w

    # cache grads next epoch
    for idx, j in enumerate(ws):
        new_prox_grad = X[:, j] @ theta
        approximate_grad = prox_grads[idx] + X[:, j] @ (hessian * X_delta_w)
        prox_grads[idx] = new_prox_grad

        prox_tol += (new_prox_grad - approximate_grad) ** 2

    # compute theta scale
    theta_scale = 1.
    max_XTtheta_ws = np.linalg.norm(prox_grads, ord=np.inf)
    if max_XTtheta_ws > alpha:
        theta_scale = alpha / max_XTtheta_ws

    return theta_scale, PROX_NEWTON_EPSILON_RATIO * prox_tol
