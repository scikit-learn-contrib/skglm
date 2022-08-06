import numpy as np
from scipy.sparse import issparse
from numba import njit

from skglm.solvers.common import construct_grad, construct_grad_sparse


def pn_solver(X, y, datafit, penalty, max_epochs=1000, w_init=None,
              max_iter=50, p0=10, tol=1e-9, use_acc=True, verbose=False):
    n_samples, n_features = X.shape
    w = np.zeros(n_features) if w is None else w_init
    Xw = np.zeros(n_samples) if Xw is None else X @ w_init
    all_features = np.arange(n_features)
    stop_crit = 0.
    obj_out = []

    is_sparse = issparse(X)
    if is_sparse:
        X_bundles = (X.data, X.indptr, X.indices)

    for t in range(max_iter):
        # compute scores
        if is_sparse:
            grad = construct_grad_sparse(*X_bundles, y, w, Xw, datafit, all_features)
        else:
            grad = construct_grad(X, y, w, Xw, datafit, all_features)

        opt = penalty.subdiff_distance(w, grad, all_features)

        # check convergences
        stop_crit = np.max(opt)
        if verbose:
            p_obj = datafit.value(y, w, Xw) + penalty.value(w)
            print(
                f"Iteration {t+1}: {p_obj:.10f}, "
                f"stopping crit: {stop_crit:.2e}"
            )

        if stop_crit <= tol:
            break

        # build working set
        gsupp_size = penalty.generalized_support(w).sum()
        ws_size = max(min(p0, n_features),
                      min(n_features, 2 * gsupp_size))
        # similar to np.argsort()[-ws_size:] but without sorting
        ws = np.argpartition(opt, -ws_size)[-ws_size:]

        for epoch in range(max_epochs):
            tol_in = 0.3 * stop_crit

            # find descent direction
            if is_sparse:
                delta_w_ws = _compute_descent_direction_s(*X_bundles, y, w, Xw,
                                                          datafit, penalty, ws,
                                                          max_cd_iter=20, tol=0.3*tol_in,
                                                          use_acc=use_acc)
            else:
                delta_w_ws = _compute_descent_direction(X, y, w, Xw, datafit, penalty,
                                                        ws, max_cd_iter=20, tol=0.3*tol_in,
                                                        use_acc=use_acc)

            # backtracking line search with inplace update of w, Xw
            if is_sparse:
                grad_ws = _backtrack_line_search_s(*X_bundles, y, w, Xw, datafit,
                                                   penalty, delta_w_ws, ws,
                                                   max_backtrack_iter=20)
            else:
                grad_ws = _backtrack_line_search(X, y, w, Xw, datafit, penalty,
                                                 delta_w_ws, ws, max_backtrack_iter=20)

            # check convergence
            opt_in = penalty.subdiff_distance(w, grad_ws, ws)
            stop_crit_in = np.max(opt_in)

            if stop_crit_in <= tol_in:
                break

        p_obj = datafit.value(y, w, Xw) + penalty.value(w)
        obj_out.append(p_obj)
    return w, obj_out, stop_crit


@njit
def _compute_descent_direction(X, y, w_epoch, Xw_epoch, datafit, penalty,
                               ws, max_cd_iter, tol, use_acc=True):
    # Given:
    #   - b = \nabla   F(X w_epoch)   <------>  raw_grad
    #   - D = \nabla^2 F(X w_epoch)   <------>  raw_hess
    # Minimize for w:
    #  b.T @ X @ (w - w_epoch) + \
    #  1/2 * (w - w_epoch).T @ X.T @ D @ X @ (w - w_epoch) + penalty(w)
    raw_grad = datafit.raw_gradient(y, Xw_epoch)
    raw_hess = datafit.raw_hessian(y, Xw_epoch, raw_grad)

    lipschitz = np.zeros(len(ws))
    for idx, j in enumerate(ws):
        lipschitz[idx] = np.sum(raw_hess * X[:, j] ** 2)

    cached_grads = np.zeros(len(ws))
    X_delta_w = np.zeros(X.shape[0])
    w_ws = w_epoch[ws]
    old_t = 1.

    for cd_iter in range(max_cd_iter):
        new_t = (1 + np.sqrt(1 + 4*old_t**2)) / 2
        for idx, j in enumerate(ws):
            cached_grads[idx] = X[:, j] @ (raw_grad + raw_hess * X_delta_w)
            old_w_idx = w_ws[idx]
            stepsize = 1 / lipschitz[idx]

            w_ws[idx] = penalty.prox_1d(
                old_w_idx - stepsize * cached_grads[idx],
                stepsize, j
            )

            # FISTA
            if use_acc:
                w_ws[idx] += (old_t - 1) / new_t * (w_ws[idx] - old_w_idx)

            if w_ws[idx] != old_w_idx:
                X_delta_w += (w_ws[idx] - old_w_idx) * X[:, j]

        if cd_iter % 5 == 0:
            opt = penalty.subdiff_distance(w_ws, cached_grads, np.arange(len(ws)))
            if np.max(opt) <= tol:
                break
        old_t = new_t

    # descent direction
    return w_ws - w_epoch[ws]


@njit
def _compute_descent_direction_s(X_data, X_indptr, X_indices, y,
                                 w_epoch, Xw_epoch, datafit, penalty,
                                 ws, max_cd_iter, tol, use_acc=True):
    # Given:
    #   - b = \nabla   F(X w_epoch)   <------>  raw_grad
    #   - D = \nabla^2 F(X w_epoch)   <------>  raw_hess
    # Minimize for w:
    #  b.T @ X @ (w - w_epoch) + \
    #  1/2 * (w - w_epoch).T @ X.T @ D @ X @ (w - w_epoch) + penalty(w)
    raw_grad = datafit.raw_gradient(y, Xw_epoch)
    raw_hess = datafit.raw_hessian(y, Xw_epoch, raw_grad)

    lipschitz = np.zeros(len(ws))
    for idx, j in enumerate(ws):
        for i in range(X_indptr[j], X_indptr[j+1]):
            lipschitz[idx] += raw_hess[X_indices[i]] * X_data[i] ** 2

    cached_grads = np.zeros(len(ws))
    X_delta_w = np.zeros(len(y))
    w_ws = w_epoch[ws]
    old_t = 1.

    for cd_iter in range(max_cd_iter):
        new_t = (1 + np.sqrt(1 + 4*old_t**2)) / 2
        for idx, j in enumerate(ws):

            # skip when the X[:, j] = 0
            if lipschitz[idx] == 0:
                continue

            cached_grads[idx] = 0.
            for i in range(X_indptr[j], X_indptr[j+1]):
                cached_grads[idx] += X_data[i] * (raw_grad[X_indices[i]] +
                                                  raw_hess[X_indices[i]] *
                                                  X_delta_w[X_indices[i]])

            old_w_idx = w_ws[idx]
            stepsize = 1 / lipschitz[idx]

            w_ws[idx] = penalty.prox_1d(
                old_w_idx - stepsize * cached_grads[idx],
                stepsize, j
            )

            # FISTA
            if use_acc:
                w_ws[idx] += (old_t - 1) / new_t * (w_ws[idx] - old_w_idx)

            if w_ws[idx] != old_w_idx:
                for i in range(X_indptr[j], X_indptr[j+1]):
                    X_delta_w[X_indices[i]] += (w_ws[idx] - old_w_idx) * X_data[i]

        if cd_iter % 5 == 0:
            opt = penalty.subdiff_distance(w_ws, cached_grads, np.arange(len(ws)))
            if np.max(opt) <= tol:
                break
        old_t = new_t

    # descent direction
    return w_ws - w_epoch[ws]


@njit
def _backtrack_line_search(X, y, w, Xw, datafit, penalty, delta_w_ws,
                           ws, max_backtrack_iter):
    # inplace update of w and Xw
    # use linear approx for diff datafit
    # return grad_ws for last w and Xw
    step, prev_step = 1., 0.
    prev_penalty_val = penalty.value(w[ws])

    for backtrack_iter in range(max_backtrack_iter):
        stop_crit = -prev_penalty_val
        for idx, j in enumerate(ws):
            old_w_j = w[j]
            w[j] += (step - prev_step) * delta_w_ws[idx]
            Xw += (w[j] - old_w_j) * X[:, j]

        grad_ws = construct_grad(X, y, w, Xw, datafit, ws)
        stop_crit += step * grad_ws @ delta_w_ws
        stop_crit += penalty.value(w[ws])

        if stop_crit < 0:
            break
        else:
            prev_step = step
            step /= 2

    return grad_ws


@njit
def _backtrack_line_search_s(X_data, X_indptr, X_indices,
                             y, w, Xw, datafit, penalty, delta_w_ws,
                             ws, max_backtrack_iter):
    # inplace update of w and Xw
    # use linear approx for diff datafit
    # return grad_ws for last w and Xw
    step, prev_step = 1., 0.
    prev_penalty_val = penalty.value(w[ws])

    for backtrack_iter in range(max_backtrack_iter):
        stop_crit = -prev_penalty_val
        for idx, j in enumerate(ws):
            old_w_j = w[j]
            w[j] += (step - prev_step) * delta_w_ws[idx]

            for i in range(X_indptr[j], X_indptr[j+1]):
                Xw[X_indices[i]] += (w[j] - old_w_j) * X_data[i]

        grad_ws = construct_grad_sparse(X_data, X_indptr, X_indices,
                                        y, w, Xw, datafit, ws)
        stop_crit += step * grad_ws.T @ delta_w_ws
        stop_crit += penalty.value(w[ws])

        if stop_crit < 0:
            break
        else:
            prev_step = step
            step /= 2

    return grad_ws
