import numpy as np
from numba import njit
from scipy.sparse import issparse

from skglm.solvers.base import BaseSolver
from skglm.utils.anderson import AndersonAcceleration
from skglm.utils.validation import check_group_compatible


class GroupBCD(BaseSolver):
    """Block coordinate descent solver for group problems.

    Attributes
    ----------
    w_init : array, shape (n_features,), default None
        Initial value of coefficients.
        If set to None, a zero vector is used instead.

    Xw_init : array, shape (n_samples,), default None
        Initial value of model fit.
        If set to None, a zero vector is used instead.

    p0 : int, default 10
        Minimum number of groups to be included in the working set.

    max_iter : int, default 1000
        Maximum number of iterations.

    max_epochs : int, default 100
        Maximum number of epochs.

    tol : float, default 1e-4
        Tolerance for convergence.

    verbose : bool, default False
        Amount of verbosity. 0/False is silent.
    """

    def __init__(self, max_iter=1000, max_epochs=100, p0=10, tol=1e-4,
                 fit_intercept=False, warm_start=False, verbose=0):
        self.max_iter = max_iter
        self.max_epochs = max_epochs
        self.p0 = p0
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.warm_start = warm_start
        self.verbose = verbose

    def solve(self, X, y, datafit, penalty, w_init=None, Xw_init=None):
        check_group_compatible(datafit)
        check_group_compatible(penalty)

        n_samples, n_features = X.shape
        n_groups = len(penalty.grp_ptr) - 1

        w = np.zeros(n_features + self.fit_intercept) if w_init is None else w_init
        Xw = np.zeros(n_samples) if w_init is None else Xw_init

        if len(w) != n_features + self.fit_intercept:
            if self.fit_intercept:
                val_error_message = (
                    "w should be of size n_features + 1 when using fit_intercept=True: "
                    f"expected {n_features + 1}, got {len(w)}.")
            else:
                val_error_message = (
                    "w should be of size n_features: "
                    f"expected {n_features}, got {len(w)}.")
            raise ValueError(val_error_message)

        is_sparse = issparse(X)
        if is_sparse:
            datafit.initialize_sparse(X.data, X.indptr, X.indices, y)
            lipschitz = datafit.get_lipschitz_sparse(X.data, X.indptr, X.indices, y)
        else:
            datafit.initialize(X, y)
            lipschitz = datafit.get_lipschitz(X, y)

        all_groups = np.arange(n_groups)
        p_objs_out = np.zeros(self.max_iter)
        stop_crit = 0.  # prevent ref before assign when max_iter == 0
        accelerator = AndersonAcceleration(K=5)

        for t in range(self.max_iter):
            if is_sparse:
                grad = _construct_grad_sparse(
                    X.data, X.indptr, X.indices, y, w, Xw, datafit, all_groups)
            else:
                grad = _construct_grad(X, y, w, Xw, datafit, all_groups)
            opt = penalty.subdiff_distance(w, grad, all_groups)

            if self.fit_intercept:
                intercept_opt = np.abs(datafit.intercept_update_step(y, Xw))
            else:
                intercept_opt = 0.

            stop_crit = max(np.max(opt), intercept_opt)

            if self.verbose:
                p_obj = datafit.value(y, w, Xw) + penalty.value(w)
                print(
                    f"Iteration {t+1}: {p_obj:.10f}, "
                    f"stopping crit: {stop_crit:.2e}"
                )

            if stop_crit <= self.tol:
                break

            gsupp_size = penalty.generalized_support(w).sum()
            ws_size = max(min(self.p0, n_groups),
                          min(n_groups, 2 * gsupp_size))
            ws = np.argpartition(opt, -ws_size)[-ws_size:]  # k-largest items (no sort)

            for epoch in range(self.max_epochs):
                # inplace update of w and Xw

                if is_sparse:
                    _bcd_epoch_sparse(X.data, X.indptr, X.indices, y,
                                      w, Xw, lipschitz, datafit, penalty, ws)

                else:
                    _bcd_epoch(X, y, w[:n_features], Xw,
                               lipschitz, datafit, penalty, ws)

                # update intercept
                if self.fit_intercept:
                    intercept_old = w[-1]
                    w[-1] -= datafit.intercept_update_step(y, Xw)
                    Xw += (w[-1] - intercept_old)

                w_acc, Xw_acc, is_extrapolated = accelerator.extrapolate(w, Xw)

                if is_extrapolated:  # avoid computing p_obj for un-extrapolated w, Xw
                    p_obj = datafit.value(y, w, Xw) + penalty.value(w)
                    p_obj_acc = datafit.value(y, w_acc, Xw_acc) + penalty.value(w_acc)

                    if p_obj_acc < p_obj:
                        w[:], Xw[:] = w_acc, Xw_acc
                        p_obj = p_obj_acc

                # check sub-optimality every 10 epochs
                if epoch % 10 == 0:
                    if is_sparse:
                        grad_ws = _construct_grad_sparse(
                            X.data, X.indptr, X.indices, y, w, Xw, datafit, ws)
                    else:
                        grad_ws = _construct_grad(X, y, w, Xw, datafit, ws)

                    opt_in = penalty.subdiff_distance(w, grad_ws, ws)
                    stop_crit_in = np.max(opt_in)

                    if max(self.verbose - 1, 0):
                        p_obj = datafit.value(y, w, Xw) + penalty.value(w)
                        print(
                            f"Epoch {epoch + 1}, objective {p_obj:.10f}, "
                            f"stopping crit {stop_crit_in:.2e}"
                        )

                    if stop_crit_in <= 0.3 * stop_crit:
                        break
            p_obj = datafit.value(y, w, Xw) + penalty.value(w)
            p_objs_out[t] = p_obj

        return w, p_objs_out, stop_crit


@njit
def _bcd_epoch(X, y, w, Xw, lipschitz, datafit, penalty, ws):
    # perform a single BCD epoch on groups in ws
    grp_ptr, grp_indices = penalty.grp_ptr, penalty.grp_indices

    for g in ws:
        grp_g_indices = grp_indices[grp_ptr[g]: grp_ptr[g+1]]
        old_w_g = w[grp_g_indices].copy()

        lipschitz_g = lipschitz[g]
        grad_g = datafit.gradient_g(X, y, w, Xw, g)

        w[grp_g_indices] = penalty.prox_1group(
            old_w_g - grad_g / lipschitz_g, 1 / lipschitz_g, g)

        for idx, j in enumerate(grp_g_indices):
            if old_w_g[idx] != w[j]:
                Xw += (w[j] - old_w_g[idx]) * X[:, j]


@njit
def _bcd_epoch_sparse(
        X_data, X_indptr, X_indices, y, w, Xw, lipschitz, datafit, penalty, ws):
    # perform a single BCD epoch on groups in ws
    grp_ptr, grp_indices = penalty.grp_ptr, penalty.grp_indices

    for g in ws:
        grp_g_indices = grp_indices[grp_ptr[g]: grp_ptr[g+1]]
        old_w_g = w[grp_g_indices].copy()

        lipschitz_g = lipschitz[g]
        grad_g = datafit.gradient_g_sparse(X_data, X_indptr, X_indices, y, w, Xw, g)

        w[grp_g_indices] = penalty.prox_1group(
            old_w_g - grad_g / lipschitz_g, 1 / lipschitz_g, g)

        for idx, j in enumerate(grp_g_indices):
            if old_w_g[idx] != w[j]:
                for i in range(X_indptr[j], X_indptr[j+1]):
                    Xw[X_indices[i]] += (w[j] - old_w_g[idx]) * X_data[i]


@njit
def _construct_grad(X, y, w, Xw, datafit, ws):
    # compute the -gradient according to each group in ws
    # note: -gradients are stacked in a 1d array ([-grad_ws_1, -grad_ws_2, ...])
    grp_ptr = datafit.grp_ptr
    n_features_ws = sum([grp_ptr[g+1] - grp_ptr[g] for g in ws])

    grads = np.zeros(n_features_ws)
    grad_ptr = 0
    for g in ws:
        grad_g = datafit.gradient_g(X, y, w, Xw, g)
        grads[grad_ptr: grad_ptr+len(grad_g)] = grad_g
        grad_ptr += len(grad_g)
    return grads


@njit
def _construct_grad_sparse(X_data, X_indptr, X_indices, y, w, Xw, datafit, ws):
    # compute the -gradient according to each group in ws
    # note: -gradients are stacked in a 1d array ([-grad_ws_1, -grad_ws_2, ...])
    grp_ptr = datafit.grp_ptr
    n_features_ws = sum([grp_ptr[g+1] - grp_ptr[g] for g in ws])

    grads = np.zeros(n_features_ws)
    grad_ptr = 0
    for g in ws:
        grad_g = datafit.gradient_g_sparse(X_data, X_indptr, X_indices, y, w, Xw, g)
        grads[grad_ptr: grad_ptr+len(grad_g)] = grad_g
        grad_ptr += len(grad_g)
    return grads
