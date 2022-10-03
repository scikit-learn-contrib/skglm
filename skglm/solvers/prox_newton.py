import numpy as np
from numba import njit
from scipy.sparse import issparse
from skglm.solvers.base import BaseSolver


EPS_TOL = 0.3
MAX_CD_ITER = 20
MAX_BACKTRACK_ITER = 20


class ProxNewton(BaseSolver):
    """Prox Newton solver combined with working sets.

    p0 : int, default 10
        Minimum number of features to be included in the working set.

    max_iter : int, default 20
        Maximum number of outer iterations.

    max_pn_iter : int, default 1000
        Maximum number of prox Newton iterations on each subproblem.

    tol : float, default 1e-4
        Tolerance for convergence.

    verbose : bool, default False
        Amount of verbosity. 0/False is silent.

    References
    ----------
    .. [1] Massias, M. and Vaiter, S. and Gramfort, A. and Salmon, J.
        "Dual Extrapolation for Sparse Generalized Linear Models", JMLR, 2020,
        https://arxiv.org/abs/1907.05830
        code: https://github.com/mathurinm/celer

    .. [2] Johnson, T. B. and Guestrin, C.
        "Blitz: A principled meta-algorithm for scaling sparse optimization",
        ICML, 2015.
        https://proceedings.mlr.press/v37/johnson15.html
        code: https://github.com/tbjohns/BlitzL1
    """

    def __init__(self, p0=10, max_iter=20, max_pn_iter=1000, tol=1e-4,
                 fit_intercept=True, warm_start=False, verbose=0):
        self.p0 = p0
        self.max_iter = max_iter
        self.max_pn_iter = max_pn_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.warm_start = warm_start
        self.verbose = verbose

    def solve(self, X, y, datafit, penalty, w_init=None, Xw_init=None):
        n_samples, n_features = X.shape
        w = np.zeros(n_features) if w_init is None else w_init
        Xw = np.zeros(n_samples) if Xw_init is None else Xw_init
        all_features = np.arange(n_features)
        stop_crit = 0.
        p_objs_out = []

        is_sparse = issparse(X)
        if is_sparse:
            X_bundles = (X.data, X.indptr, X.indices)

        for t in range(self.max_iter):
            # compute scores
            if is_sparse:
                grad = _construct_grad_sparse(
                    *X_bundles, y, w, Xw, datafit, all_features)
            else:
                grad = _construct_grad(X, y, w, Xw, datafit, all_features)

            opt = penalty.subdiff_distance(w, grad, all_features)

            # check convergences
            stop_crit = np.max(opt)
            if self.verbose:
                p_obj = datafit.value(y, w, Xw) + penalty.value(w)
                print(
                    "Iteration {}: {:.10f}, ".format(t+1, p_obj) +
                    "stopping crit: {:.2e}".format(stop_crit)
                )

            if stop_crit <= self.tol:
                if self.verbose:
                    print("Stopping criterion max violation: {:.2e}".format(stop_crit))
                break

            # build working set
            gsupp_size = penalty.generalized_support(w).sum()
            ws_size = max(min(self.p0, n_features),
                          min(n_features, 2 * gsupp_size))
            # similar to np.argsort()[-ws_size:] but without sorting
            ws = np.argpartition(opt, -ws_size)[-ws_size:]

            grad_ws = grad[ws]
            tol_in = EPS_TOL * stop_crit

            for pn_iter in range(self.max_pn_iter):
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
                    grad_ws[:] = _backtrack_line_search_s(
                        *X_bundles, y, w, Xw, datafit, penalty, delta_w_ws,
                        X_delta_w_ws, ws)
                else:
                    grad_ws[:] = _backtrack_line_search(
                        X, y, w, Xw, datafit, penalty, delta_w_ws, X_delta_w_ws, ws)

                # check convergence
                opt_in = penalty.subdiff_distance(w, grad_ws, ws)
                stop_crit_in = np.max(opt_in)

                if max(self.verbose-1, 0):
                    p_obj = datafit.value(y, w, Xw) + penalty.value(w)
                    print(
                        "PN iteration {}: {:.10f}, ".format(pn_iter+1, p_obj) +
                        "stopping crit in: {:.2e}".format(stop_crit_in)
                    )

                if stop_crit_in <= tol_in:
                    if max(self.verbose-1, 0):
                        print("Early exit")
                    break

            p_obj = datafit.value(y, w, Xw) + penalty.value(w)
            p_objs_out.append(p_obj)
        return w, np.asarray(p_objs_out), stop_crit


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

    # for a less costly stopping criterion, we do not compute the exact gradient,
    # but store each coordinate-wise gradient every time we update one coordinate:
    past_grads = np.zeros(len(ws))
    X_delta_w_ws = np.zeros(X.shape[0])
    w_ws = w_epoch[ws]

    for cd_iter in range(MAX_CD_ITER):
        for idx, j in enumerate(ws):
            # skip when X[:, j] == 0
            if lipschitz[idx] == 0:
                continue

            past_grads[idx] = grad_ws[idx] + X[:, j] @ (raw_hess * X_delta_w_ws)
            old_w_idx = w_ws[idx]
            stepsize = 1 / lipschitz[idx]

            w_ws[idx] = penalty.prox_1d(
                old_w_idx - stepsize * past_grads[idx], stepsize, j)

            if w_ws[idx] != old_w_idx:
                X_delta_w_ws += (w_ws[idx] - old_w_idx) * X[:, j]

        if cd_iter % 5 == 0:
            # TODO: can be improved by passing in w_ws but breaks for WeightedL1
            current_w = w_epoch.copy()
            current_w[ws] = w_ws
            opt = penalty.subdiff_distance(current_w, past_grads, ws)
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
        lipschitz[idx] = _sparse_squared_weighted_norm(
            X_data, X_indptr, X_indices, j, raw_hess)

    # see _descent_direction() comment
    past_grads = np.zeros(len(ws))
    X_delta_w_ws = np.zeros(len(y))
    w_ws = w_epoch[ws]

    for cd_iter in range(MAX_CD_ITER):
        for idx, j in enumerate(ws):
            # skip when X[:, j] == 0
            if lipschitz[idx] == 0:
                continue

            past_grads[idx] = grad_ws[idx]
            # equivalent to cached_grads[idx] += X[:, j] @ (raw_hess * X_delta_w_ws)
            past_grads[idx] += _sparse_weighted_dot(
                X_data, X_indptr, X_indices, j, X_delta_w_ws, raw_hess)

            old_w_idx = w_ws[idx]
            stepsize = 1 / lipschitz[idx]

            w_ws[idx] = penalty.prox_1d(
                old_w_idx - stepsize * past_grads[idx], stepsize, j)

            if w_ws[idx] != old_w_idx:
                _update_X_delta_w(X_data, X_indptr, X_indices, X_delta_w_ws,
                                  w_ws[idx] - old_w_idx, j)

        if cd_iter % 5 == 0:
            # TODO: could be improved by passing in w_ws
            current_w = w_epoch.copy()
            current_w[ws] = w_ws
            opt = penalty.subdiff_distance(current_w, past_grads, ws)
            if np.max(opt) <= tol:
                break

    # descent direction
    return w_ws - w_epoch[ws], X_delta_w_ws


@njit
def _backtrack_line_search(X, y, w, Xw, datafit, penalty, delta_w_ws,
                           X_delta_w_ws, ws):
    # 1) find step in [0, 1] such that:
    #   penalty(w + step * delta_w) - penalty(w) +
    #   step * \nabla datafit(w + step * delta_w) @ delta_w < 0
    # ref: https://www.di.ens.fr/~aspremon/PDF/ENSAE/Newton.pdf
    # 2) inplace update of w and Xw and return grad_ws of the last w and Xw
    step, prev_step = 1., 0.
    # TODO: could be improved by passing in w[ws]
    old_penalty_val = penalty.value(w)

    # try step = 1, 1/2, 1/4, ...
    for _ in range(MAX_BACKTRACK_ITER):
        w[ws] += (step - prev_step) * delta_w_ws
        Xw += (step - prev_step) * X_delta_w_ws

        grad_ws = _construct_grad(X, y, w, Xw, datafit, ws)
        # TODO: could be improved by passing in w[ws]
        stop_crit = penalty.value(w) - old_penalty_val
        stop_crit += step * grad_ws @ delta_w_ws

        if stop_crit < 0:
            break
        else:
            prev_step = step
            step /= 2
    else:
        pass
        # TODO this case is not handled yet

    return grad_ws


# sparse version of _backtrack_line_search
@njit
def _backtrack_line_search_s(X_data, X_indptr, X_indices, y, w, Xw, datafit,
                             penalty, delta_w_ws, X_delta_w_ws, ws):
    step, prev_step = 1., 0.
    # TODO: could be improved by passing in w[ws]
    old_penalty_val = penalty.value(w)

    for _ in range(MAX_BACKTRACK_ITER):
        w[ws] += (step - prev_step) * delta_w_ws
        Xw += (step - prev_step) * X_delta_w_ws

        grad_ws = _construct_grad_sparse(X_data, X_indptr, X_indices,
                                         y, w, Xw, datafit, ws)
        # TODO: could be improved by passing in w[ws]
        stop_crit = penalty.value(w) - old_penalty_val
        stop_crit += step * grad_ws.T @ delta_w_ws

        if stop_crit < 0:
            break
        else:
            prev_step = step
            step /= 2
    else:
        pass  # TODO

    return grad_ws


@njit
def _construct_grad(X, y, w, Xw, datafit, ws):
    # Compute grad of datafit restricted to ws. This function avoids
    # recomputing raw_grad for every j, which is costly for logreg
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
