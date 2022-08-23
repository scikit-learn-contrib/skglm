import numpy as np
from numba import njit
from scipy.sparse import issparse


EPS_TOL = 0.3
MAX_CD_ITER = 20
MAX_BACKTRACK_ITER = 20


def prox_newton(X, y, datafit, penalty, w_init=None, p0=10,
                max_iter=20, max_epochs=1000, tol=1e-4, verbose=0):
    """Run a Prox Newton solver.

    Parameters
    ----------
    X : array or sparse CSC matrix, shape (n_samples, n_features)
        Design matrix.

    y : array, shape (n_samples,)
        Target vector.

    datafit : instance of BaseDatafit
        Datafit object.

    penalty : instance of BasePenalty
        Penalty object.

    w_init : array, shape (n_features,), default None
        Initial value of coefficients.
        If set to None, a zero vector is used instead.

    p0 : int, default 10
        Minimum number of features to be included in the working set.

    max_iter : int, default 20
        Maximum number of outer iterations.

    max_epochs : int, default 1000
        Maximum number of epochs.

    tol : float, default 1e-4
        Tolerance for convergence.

    verbose : bool, default False
        Amount of verbosity. 0/False is silent.

    Returns
    -------
    w : array, shape (n_features,)
        Solution that minimizes the problem defined by datafit and penalty.

    objs_out: array (max_iter,)
        The objective values at every outer iteration.

    stop_crit: float
        The value of the stopping criterion when the solver stops.
    """
    n_samples, n_features = X.shape
    w = np.zeros(n_features) if w_init is None else w_init
    Xw = np.zeros(n_samples) if w_init is None else X @ w_init
    all_features = np.arange(n_features)
    stop_crit = 0.
    p_objs_out = []

    is_sparse = issparse(X)
    if is_sparse:
        X_bundles = (X.data, X.indptr, X.indices)

    for t in range(max_iter):
        # compute scores
        if is_sparse:
            grad = _construct_grad_sparse(*X_bundles, y, w, Xw, datafit, all_features)
        else:
            grad = _construct_grad(X, y, w, Xw, datafit, all_features)

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
            if verbose:
                print(f"Stopping criterion max violation: {stop_crit:.2e}")
            break

        # build working set
        gsupp_size = penalty.generalized_support(w).sum()
        ws_size = max(min(p0, n_features),
                      min(n_features, 2 * gsupp_size))
        # similar to np.argsort()[-ws_size:] but without sorting
        ws = np.argpartition(opt, -ws_size)[-ws_size:]

        grad_ws = grad[ws]
        tol_in = EPS_TOL * stop_crit

        for epoch in range(max_epochs):

            # find descent direction
            if is_sparse:
                delta_w_ws, X_delta_w_ws = _descent_direction_s(
                    *X_bundles, y, w, Xw, grad_ws, datafit,
                    penalty, ws, tol=EPS_TOL*tol_in)
            else:
                delta_w_ws, X_delta_w_ws = _descent_direction(
                    X, y, w, Xw, grad_ws, datafit, penalty, ws, tol=EPS_TOL*tol_in)

            # backtracking line search with inplace update of w, Xw
            if is_sparse:
                grad_ws[:] = _backtrack_line_search_s(*X_bundles, y, w, Xw,
                                                      datafit, penalty, delta_w_ws,
                                                      X_delta_w_ws, ws)
            else:
                grad_ws[:] = _backtrack_line_search(X, y, w, Xw, datafit, penalty,
                                                    delta_w_ws, X_delta_w_ws, ws)

            # check convergence
            opt_in = penalty.subdiff_distance(w, grad_ws, ws)
            stop_crit_in = np.max(opt_in)

            if max(verbose-1, 0):
                p_obj = datafit.value(y, w, Xw) + penalty.value(w)
                print(
                    f"Epoch {epoch+1}: {p_obj:.10f}, "
                    f"stopping crit in: {stop_crit_in:.2e}"
                )

            if stop_crit_in <= tol_in:
                if max(verbose-1, 0):
                    print("Early exit")
                break

        p_obj = datafit.value(y, w, Xw) + penalty.value(w)
        p_objs_out.append(p_obj)
    return w, p_objs_out, stop_crit


@njit
def _descent_direction(X, y, w_epoch, Xw_epoch, grad_ws, datafit,
                       penalty, ws, tol):
    # Given:
    #   1) b = \nabla F(X w_epoch)
    #   2) D = \nabla^2 F(X w_epoch)  <------>  raw_hess
    # Minimize quadratic approximation for delta_w = w - w_epoch:
    #  b.T @ X @ delta_w + \
    #  1/2 * delta_w.T @ (X.T @ D @ X) @ delta_w + penalty(w)
    raw_hess = datafit.raw_hessian(y, Xw_epoch)

    lipschitz = np.zeros(len(ws))
    for idx, j in enumerate(ws):
        lipschitz[idx] = raw_hess @ X[:, j] ** 2

    cached_grads = np.zeros(len(ws))
    X_delta_w_ws = np.zeros(X.shape[0])
    w_ws = w_epoch[ws]

    for cd_iter in range(MAX_CD_ITER):
        for idx, j in enumerate(ws):
            # skip when X[:, j] == 0
            if lipschitz[idx] == 0:
                continue

            cached_grads[idx] = grad_ws[idx] + X[:, j] @ (raw_hess * X_delta_w_ws)
            old_w_idx = w_ws[idx]
            stepsize = 1 / lipschitz[idx]

            w_ws[idx] = penalty.prox_1d(
                old_w_idx - stepsize * cached_grads[idx], stepsize, j)

            if w_ws[idx] != old_w_idx:
                X_delta_w_ws += (w_ws[idx] - old_w_idx) * X[:, j]

        if cd_iter % 5 == 0:
            opt = penalty.subdiff_distance(w_ws, cached_grads, ws)
            if np.max(opt) <= tol:
                break

    # descent direction
    return w_ws - w_epoch[ws], X_delta_w_ws


# sparse version of _compute_descent_direction
@njit
def _descent_direction_s(X_data, X_indptr, X_indices, y, w_epoch,
                         Xw_epoch, grad_ws, datafit, penalty, ws, tol):
    raw_hess = datafit.raw_hessian(y, Xw_epoch)

    lipschitz = np.zeros(len(ws))
    for idx, j in enumerate(ws):
        # equivalent to: lipschitz[idx] += raw_hess * X[:, j] ** 2
        lipschitz[idx] = _sparse_squared_weighted_norm(X_data, X_indptr, X_indices,
                                                       j, raw_hess)

    cached_grads = np.zeros(len(ws))
    X_delta_w_ws = np.zeros(len(y))
    w_ws = w_epoch[ws]

    for cd_iter in range(MAX_CD_ITER):
        for idx, j in enumerate(ws):
            # skip when X[:, j] == 0
            if lipschitz[idx] == 0:
                continue

            cached_grads[idx] = grad_ws[idx]
            # equivalent to cached_grads[idx] += X[:, j] @ (raw_hess * X_delta_w_ws)
            cached_grads[idx] += _sparse_weighted_dot(X_data, X_indptr, X_indices, j,
                                                      X_delta_w_ws, raw_hess)

            old_w_idx = w_ws[idx]
            stepsize = 1 / lipschitz[idx]

            w_ws[idx] = penalty.prox_1d(
                old_w_idx - stepsize * cached_grads[idx], stepsize, j)

            if w_ws[idx] != old_w_idx:
                _update_X_delta_w(X_data, X_indptr, X_indices, X_delta_w_ws,
                                  w_ws[idx] - old_w_idx, j)

        if cd_iter % 5 == 0:
            opt = penalty.subdiff_distance(w_ws, cached_grads, ws)
            if np.max(opt) <= tol:
                break

    # descent direction
    return w_ws - w_epoch[ws], X_delta_w_ws


@njit
def _backtrack_line_search(X, y, w, Xw, datafit, penalty, delta_w_ws,
                           X_delta_w_ws, ws):
    # inplace update of w and Xw
    # return grad_ws of the last w and Xw
    step, prev_step = 1., 0.
    epoch_penalty_val = penalty.value(w[ws])

    for backtrack_iter in range(MAX_BACKTRACK_ITER):
        stop_crit = -epoch_penalty_val
        w[ws] += (step - prev_step) * delta_w_ws
        Xw += (step - prev_step) * X_delta_w_ws

        grad_ws = _construct_grad(X, y, w, Xw, datafit, ws)
        stop_crit += step * grad_ws @ delta_w_ws
        stop_crit += penalty.value(w[ws])

        if stop_crit < 0:
            break
        else:
            prev_step = step
            step /= 2

    return grad_ws


# sparse version of _backtrack_line_search
@njit
def _backtrack_line_search_s(X_data, X_indptr, X_indices, y, w, Xw, datafit,
                             penalty, delta_w_ws, X_delta_w_ws, ws):
    # inplace update of w and Xw
    step, prev_step = 1., 0.
    epoch_penalty_val = penalty.value(w[ws])

    for backtrack_iter in range(MAX_BACKTRACK_ITER):
        stop_crit = -epoch_penalty_val
        w[ws] += (step - prev_step) * delta_w_ws
        Xw += (step - prev_step) * X_delta_w_ws

        grad_ws = _construct_grad_sparse(X_data, X_indptr, X_indices,
                                         y, w, Xw, datafit, ws)
        stop_crit += step * grad_ws.T @ delta_w_ws
        stop_crit += penalty.value(w[ws])

        if stop_crit < 0:
            break
        else:
            prev_step = step
            step /= 2

    return grad_ws


@njit
def _construct_grad(X, y, w, Xw, datafit, ws):
    # Compute grad of datafit restricted to ws
    raw_grad = datafit.raw_grad(y, Xw)
    grad = np.zeros(len(ws))
    for idx, j in enumerate(ws):
        grad[idx] = X[:, j] @ raw_grad
    return grad


@njit
def _construct_grad_sparse(X_data, X_indptr, X_indices, y, w, Xw, datafit, ws):
    # Compute grad of datafit restricted to ws in case X sparse
    raw_grad = datafit.raw_grad(y, Xw)
    grad = np.zeros(len(ws))
    for idx, j in enumerate(ws):
        grad[idx] = _sparse_xj_dot(X_data, X_indptr, X_indices, j, raw_grad)
    return grad


@njit(fastmath=True)
def _sparse_xj_dot(X_data, X_indptr, X_indices, j, other):
    # Compute X[:, j] @ other in case X sparse
    res = 0.
    for i in range(X_indptr[j], X_indptr[j+1]):
        res += X_data[i] * other[X_indices[i]]
    return res


@njit(fastmath=True)
def _sparse_weighted_dot(X_data, X_indptr, X_indices, j, other, weights):
    # Compute X[:, j] @ (weights * other) in case X sparse
    res = 0.
    for i in range(X_indptr[j], X_indptr[j+1]):
        res += X_data[i] * other[X_indices[i]] * weights[X_indices[i]]
    return res


@njit(fastmath=True)
def _sparse_squared_weighted_norm(X_data, X_indptr, X_indices, j, weights):
    # Compute weights @ X[:, j]**2 in case X sparse
    res = 0.
    for i in range(X_indptr[j], X_indptr[j+1]):
        res += weights[X_indices[i]] * X_data[i]**2
    return res


@njit(fastmath=True)
def _update_X_delta_w(X_data, X_indptr, X_indices, X_delta_w, diff, j):
    # Compute X_delta_w += diff * X[:, j] in case of X sparse
    for i in range(X_indptr[j], X_indptr[j+1]):
        X_delta_w[X_indices[i]] += diff * X_data[i]
