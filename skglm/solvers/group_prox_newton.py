import numpy as np
from numba import njit
from numpy.linalg import norm
from skglm.solvers.base import BaseSolver
from skglm.utils import check_group_compatible

EPS_TOL = 0.3
MAX_CD_ITER = 20
MAX_BACKTRACK_ITER = 20


class GroupProxNewton(BaseSolver):
    """Group Prox Newton solver combined with working sets.

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
                 fit_intercept=False, warm_start=False, verbose=0):
        self.p0 = p0
        self.max_iter = max_iter
        self.max_pn_iter = max_pn_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.warm_start = warm_start
        self.verbose = verbose

    def solve(self, X, y, datafit, penalty, w_init=None, Xw_init=None):
        check_group_compatible(datafit)
        check_group_compatible(penalty)

        n_samples, n_features = X.shape
        grp_ptr, grp_indices = penalty.grp_ptr, penalty.grp_indices
        n_groups = len(grp_ptr) - 1

        w = np.zeros(n_features) if w_init is None else w_init
        Xw = np.zeros(n_samples) if Xw_init is None else Xw_init
        all_groups = np.arange(n_groups)
        stop_crit = 0.
        p_objs_out = []

        for iter in range(self.max_iter):
            grad = -_construct_grad(X, y, w, Xw, datafit, all_groups)

            # check convergence
            opt = penalty.subdiff_distance(w, -grad, all_groups)
            stop_crit = np.max(opt)

            if self.verbose:
                p_obj = datafit.value(y, w, Xw) + penalty.value(w)
                print(
                    f"Iteration {iter+1}: {p_obj:.10f}, "
                    f"stopping crit: {stop_crit:.2e}"
                )

            if stop_crit <= self.tol:
                break

            # build working set ws
            gsupp_size = penalty.generalized_support(w).sum()
            ws_size = max(min(self.p0, n_groups),
                          min(n_groups, 2 * gsupp_size))
            ws = np.argpartition(opt, -ws_size)[-ws_size:]  # k-largest items (no sort)

            grad_ws = _slice_array(grad, ws, grp_ptr, grp_indices)
            tol_in = EPS_TOL * stop_crit

            # solve subproblem restricted to ws
            for pn_iter in range(self.max_pn_iter):
                # find descent direction
                delta_w_ws, X_delta_w_ws = _descent_direction(
                    X, y, w, Xw, grad_ws, datafit, penalty, ws, tol=EPS_TOL*tol_in)

                # find a suitable step size and in-place update w, Xw
                grad_ws[:] = _backtrack_line_search(
                    X, y, w, Xw, datafit, penalty, delta_w_ws, X_delta_w_ws, ws)

                # check convergence
                opt_in = penalty.subdiff_distance(w, -grad_ws, ws)
                stop_crit_in = np.max(opt_in)

                if max(self.verbose-1, 0):
                    p_obj = datafit.value(y, w, Xw) + penalty.value(w)
                    print(
                        f"PN iteration {pn_iter+1,}: {p_obj:.10f}, "
                        f"stopping crit in: {stop_crit_in:.2e}"
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
    # given:
    #   1) b = \nabla F(X w_epoch)
    #   2) D = \nabla^2 F(X w_epoch)  <------>  raw_hess
    # minimize quadratic approximation for delta_w = w - w_epoch:
    #  b.T @ X @ delta_w + \
    #  1/2 * delta_w.T @ (X.T @ D @ X) @ delta_w + penalty(w)
    # In BCD, we leverage inequality:
    #  penalty_g(w_g) + 1/2 ||delta_w_g||_H <= \
    #  penalty_g(w_g) + 1/2 * || H || * ||delta_w_g||
    grp_ptr, grp_indices = penalty.grp_ptr, penalty.grp_indices
    n_features_ws = sum([penalty.grp_ptr[g+1] - penalty.grp_ptr[g] for g in ws])
    raw_hess = datafit.raw_hessian(y, Xw_epoch)

    lipchitz = np.zeros(len(ws))
    for idx, g in enumerate(ws):
        grp_g_indices = grp_indices[grp_ptr[g]:grp_ptr[g+1]]
        # equivalent to: norm(X[:, grp_g_indices].T @ D @ X[:, grp_g_indices], ord=2)
        lipchitz[idx] = norm(_diag_times_X_g(
            np.sqrt(raw_hess), X, grp_g_indices), ord=2)**2

    # for a less costly stopping criterion, we do no compute the exact gradient,
    # but store each coordinate-wise gradient every time we update one coordinate:
    past_grads = np.zeros(n_features_ws)
    X_delta_w_ws = np.zeros(X.shape[0])
    w_ws = _slice_array(w_epoch, ws, grp_ptr, grp_indices)

    for cd_iter in range(MAX_CD_ITER):
        ptr = 0
        for idx, g in enumerate(ws):
            # skip when X[:, grp_g_indices] == 0
            if lipchitz[idx] == 0.:
                continue

            grp_g_indices = grp_indices[grp_ptr[g]:grp_ptr[g+1]]
            range_grp_g = slice(ptr, ptr + len(grp_g_indices))

            past_grads[range_grp_g] = grad_ws[range_grp_g]
            # += X[:, grp_g_indices].T @ (raw_hess * X_delta_w_ws)
            past_grads[range_grp_g] += _X_g_T_dot_vec(
                X, raw_hess * X_delta_w_ws, grp_g_indices)

            old_w_ws_g = w_ws[range_grp_g].copy()
            stepsize = 1 / lipchitz[idx]

            w_ws[range_grp_g] = penalty.prox_1group(
                old_w_ws_g - stepsize * past_grads[range_grp_g], stepsize, g)

            # X_delta_w_ws += X[:, grp_g_indices] @ (w_ws[range_grp_g] - old_w_ws_g)
            _update_X_delta_w_ws(X, X_delta_w_ws, w_ws[range_grp_g], old_w_ws_g,
                                 grp_g_indices)

            ptr += len(grp_g_indices)

        if cd_iter % 5 == 0:
            # TODO: can be improved by passing in w_ws
            current_w = w_epoch.copy()

            # current_w[ws] = w_ws
            ptr = 0
            for g in ws:
                grp_g_indices = grp_indices[grp_ptr[g]:grp_ptr[g+1]]
                current_w[grp_g_indices] = w_ws[ptr: ptr+len(grp_g_indices)]
                ptr += len(grp_g_indices)

            opt = penalty.subdiff_distance(current_w, past_grads, ws)
            if np.max(opt) <= tol:
                break

    # descent direction
    delta_w_ws = w_ws - _slice_array(w_epoch, ws, grp_ptr, grp_indices)
    return delta_w_ws, X_delta_w_ws


@njit
def _backtrack_line_search(X, y, w, Xw, datafit, penalty, delta_w_ws,
                           X_delta_w_ws, ws):
    # 1) find step in [0, 1] such that:
    #   penalty(w + step * delta_w) - penalty(w) +
    #   step * \nabla datafit(w + step * delta_w) @ delta_w < 0
    # ref: https://www.di.ens.fr/~aspremon/PDF/ENSAE/Newton.pdf
    # 2) inplace update of w and Xw and return grad_ws of the last w and Xw
    grp_ptr, grp_indices = penalty.grp_ptr, penalty.grp_indices
    step, prev_step = 1., 0.

    # TODO: could be improved by passing in w[ws]
    old_penalty_val = penalty.value(w)

    # try step = 1, 1/2, 1/4, ...
    for _ in range(MAX_BACKTRACK_ITER):
        # w[ws] += (step - prev_step) * delta_w_ws
        ptr = 0
        for g in ws:
            grp_g_indices = grp_indices[grp_ptr[g]:grp_ptr[g+1]]
            w[grp_g_indices] += ((step - prev_step) *
                                 delta_w_ws[ptr: ptr + len(grp_g_indices)])
            ptr += len(grp_g_indices)

        Xw += (step - prev_step) * X_delta_w_ws
        grad_ws = -_construct_grad(X, y, w, Xw, datafit, ws)

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


@njit
def _construct_grad(X, y, w, Xw, datafit, ws):
    # compute grad of datafit restricted to ws. This function avoids
    # recomputing raw_grad for every j, which is costly for logreg
    grp_ptr, grp_indices = datafit.grp_ptr, datafit.grp_indices
    n_features_ws = sum([grp_ptr[g+1] - grp_ptr[g] for g in ws])

    raw_grad = datafit.raw_grad(y, Xw)
    minus_grad = np.zeros(n_features_ws)

    ptr = 0
    for g in ws:
        # compute grad_g
        grp_g_indices = grp_indices[grp_ptr[g]:grp_ptr[g+1]]
        for j in grp_g_indices:
            minus_grad[ptr] = -X[:, j] @ raw_grad
            ptr += 1

    return minus_grad


@njit
def _slice_array(arr, ws, grp_ptr, grp_indices):
    # returns [arr[ws_1], arr[ws_2], ...]
    n_features_ws = sum([grp_ptr[g+1] - grp_ptr[g] for g in ws])
    sliced_arr = np.zeros(n_features_ws)

    ptr = 0
    for g in ws:
        grp_g_indices = grp_indices[grp_ptr[g]:grp_ptr[g+1]]
        sliced_arr[ptr: ptr+len(grp_g_indices)] = arr[grp_g_indices]
        ptr += len(grp_g_indices)

    return sliced_arr


@njit
def _update_X_delta_w_ws(X, X_delta_w_ws, w_ws_g, old_w_ws_g, grp_g_indices):
    #
    for idx, j in enumerate(grp_g_indices):
        delta_w_j = w_ws_g[idx] - old_w_ws_g[idx]
        if w_ws_g[idx] != old_w_ws_g[idx]:
            X_delta_w_ws += delta_w_j * X[:, j]


@njit
def _X_g_T_dot_vec(X, vec, grp_g_indices):
    #
    result = np.zeros(len(grp_g_indices))
    for idx, j in enumerate(grp_g_indices):
        result[idx] = X[:, j] @ vec
    return result


@njit
def _diag_times_X_g(diag, X, grp_g_indices):
    #
    result = np.zeros((len(diag), len(grp_g_indices)))
    for idx, j in enumerate(grp_g_indices):
        result[:, idx] = diag * X[:, j]
    return result
