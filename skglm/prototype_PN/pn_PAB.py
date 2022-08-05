import numpy as np
from scipy import sparse
from numba import njit

from skglm.datafits.single_task import sigmoid
from skglm.solvers.common import construct_grad


def prox_newton_solver(
        X, y, datafit, penalty, w=None, Xw=None, max_iter=50, max_epochs=1000, max_backtrack=20,
        min_pn_cd_epochs=2, max_pn_cd_epochs=20, p0=10, tol=1e-4, verbose=0,
        eps_in=0.3):

    n_samples, n_features = X.shape
    pen = penalty.is_penalized(n_features)
    unpen = ~pen
    n_unpen = unpen.sum()
    obj_out = []
    all_feats = np.arange(n_features)
    stop_crit = np.inf  # initialize for case n_iter=0
    w = np.zeros(n_features) if w is None else w
    Xw = np.zeros(n_samples) if Xw is None else Xw

    hessian_diag = np.zeros(n_samples)  # hessian = X^T D X, with D = diag(f_i'')
    grad_datafit = np.zeros(n_samples)  # gradient of F(Xw)

    is_sparse = sparse.issparse(X)
    if is_sparse:
        X_bundles = (X.data, X.indptr, X.indices)

    for t in range(max_iter):
        if is_sparse:
            grad = datafit.full_grad_sparse(*X_bundles, y, Xw)
        else:
            grad = construct_grad(X, y, w, Xw, datafit, all_feats)

        # TODO: support fix point criterion
        opt = penalty.subdiff_distance(w, grad, all_feats)
        stop_crit = np.max(opt)
        if verbose:
            print(f"Stopping criterion max violation: {stop_crit:.2e}")
        if stop_crit <= tol:
            break
        # 1) select features: all unpenalized + 2 * (nnz and penalized)
        ws_size = max(min(p0 + n_unpen, n_features),
                      min(2 * penalty.generalized_support(w).sum() -
                          n_unpen, n_features))

        opt[unpen] = np.inf  # always include unpenalized features
        opt[penalty.generalized_support(w)] = np.inf

        # taking arg_top_K features violating the stopping criterion
        if ws_size == n_features:
            ws = np.arange(n_features)
        else:
            ws = np.argpartition(opt, -ws_size)[-ws_size:]

        if verbose:
            print(f"Iteration {t + 1}, {ws_size} feats in subpb.")

        lc = np.zeros(ws_size)  # weighted Lipschitz constants
        old_grad = np.zeros(ws_size)
        pn_grad_diff = 0.
        pn_tol_ratio = 10.

        if is_sparse:
            _compute_grad_hessian_datafit_s(*X_bundles, y, Xw, ws, 
                                            hessian_diag, grad_datafit, lc)
        else:
            _compute_grad_hessian_datafit(X, y, Xw, ws, hessian_diag, grad_datafit, lc)

        # 2) run prox newton on smaller subproblem
        for epoch in range(max_epochs):
            # TODO: support sparse matrices
            # pn_grad_diff, n_performed_cd_epochs = _prox_newton_iter(
            #     X, Xw, w, y, penalty, ws, min_pn_cd_epochs, max_pn_cd_epochs,
            #     max_backtrack, pn_tol_ratio, pn_grad_diff, hessian_diag, grad_datafit,
            #     lc, old_grad, epoch, verbose=verbose)
            if is_sparse:
                delta_w, X_delta_w, n_performed_cd_epochs = _compute_descent_direction_s(
                    *X_bundles, w, ws, hessian_diag, old_grad, grad_datafit, lc, penalty, min_pn_cd_epochs,
                    max_pn_cd_epochs, pn_tol_ratio, pn_grad_diff, epoch, verbose=verbose)
            else:
                delta_w, X_delta_w, n_performed_cd_epochs = _compute_descent_direction(
                    X, w, ws, hessian_diag, old_grad, grad_datafit, lc, penalty, min_pn_cd_epochs,
                    max_pn_cd_epochs, pn_tol_ratio, pn_grad_diff, epoch, verbose=verbose)


            _backtrack_line_search(w, Xw, delta_w, X_delta_w,
                                   ws, y, penalty, max_backtrack)

            if is_sparse:
                _compute_grad_hessian_datafit_s(*X_bundles, y, Xw, ws, 
                                            hessian_diag, grad_datafit, lc)
            else:
                _compute_grad_hessian_datafit(X, y, Xw, ws, hessian_diag, grad_datafit, lc)

            pn_grad_diff = 0.
            for idx, j in enumerate(ws):
                if is_sparse:
                    tmp = 0.
                    for i in range(X.indptr[j], X.indptr[j+1]):
                        tmp += X.data[i] * grad_datafit[X.indices[i]]
                    actual_grad = tmp

                    tmp = 0.
                    for i in range(X.indptr[j], X.indptr[j+1]):
                        tmp += X.data[i] * X_delta_w[X.indices[i]] * hessian_diag[X.indices[i]]
                    approx_grad = old_grad[idx] + tmp
                else:
                    actual_grad = X[:, j] @ grad_datafit
                    approx_grad = old_grad[idx] + (X[:, j] * X_delta_w * hessian_diag).sum()

                old_grad[idx] = actual_grad
                grad_diff = actual_grad - approx_grad
                pn_grad_diff += grad_diff ** 2

            p_obj = datafit.value(y, w, Xw) + penalty.value(w)

            if is_sparse:
                grad = datafit.full_grad_sparse(*X_bundles, y, Xw)
            else:
                grad = construct_grad(X, y, w, Xw, datafit, ws)

            opt_ws = penalty.subdiff_distance(w, grad, ws)
            stop_crit_in = np.max(opt_ws)
            tol_in = eps_in * stop_crit
            if stop_crit_in <= tol_in:
                if max(verbose - 1, 0):
                    print("Early exit")
                break
        obj_out.append(p_obj)
    return w, np.array(obj_out), stop_crit


@njit
def _compute_descent_direction(
        X, w, ws, hessian_diag, old_grad, grad_datafit, lc, penalty, min_pn_cd_epochs,
        max_pn_cd_epochs, pn_tol_ratio, pn_grad_diff, epoch, verbose=0):
    delta_w, X_delta_w = np.zeros(len(ws)), np.zeros(X.shape[0])
    pn_tol = 0.
    _max_pn_cd_epochs = max_pn_cd_epochs
    if epoch == 0:
        _max_pn_cd_epochs = min_pn_cd_epochs
        for idx, j in enumerate(ws):
            old_grad[idx] = X[:, j] @ grad_datafit
    else:
        pn_tol = pn_tol_ratio * pn_grad_diff

    n_performed_cd_epochs = 0
    for pn_cd_epoch in range(_max_pn_cd_epochs):
        weighted_fix_point_crit = 0.
        n_performed_cd_epochs += 1
        for idx, j in enumerate(ws):
            stepsize = 1/lc[idx] if lc[idx] != 0 else 1000
            old_value = w[j] + delta_w[idx]
            # TODO: Is it the correct gradient? What's the intuition behind the formula?
            grad_j = old_grad[idx] + (X[:, j] * X_delta_w * hessian_diag).sum()
            new_value = penalty.prox_1d(
                old_value - grad_j * stepsize, stepsize, j)
            diff = new_value - old_value
            if diff != 0:
                weighted_fix_point_crit += (diff * lc[idx]) ** 2
                delta_w[idx] = new_value - w[j]
                for i in range(X.shape[0]):
                    X_delta_w[i] += diff * X[i, j]
            # if max(verbose - 1, 0):
            #     print("delta w is ", delta_w[idx])
        # weighted_fix_point_crit /= X.shape[0] ** 2
        # TODO: beware scaling weighted_fix_point_crit and pn_tol
        # try a more proncipled criterion? subdiff dist?
        if weighted_fix_point_crit / X.shape[0] ** 2 <= pn_tol and pn_cd_epoch + 1 >= min_pn_cd_epochs:
            if max(verbose - 1, 0):
                print("Exited! weighted_fix_point_crit: ", weighted_fix_point_crit)
            break
    return delta_w, X_delta_w, n_performed_cd_epochs



@njit
def _compute_descent_direction_s(
        X_data, X_indptr, X_indices, w, ws, hessian_diag, old_grad, 
        grad_datafit, lc, penalty, min_pn_cd_epochs,
        max_pn_cd_epochs, pn_tol_ratio, pn_grad_diff, epoch, verbose=0):
    delta_w, X_delta_w = np.zeros(len(ws)), np.zeros(len(hessian_diag))
    pn_tol = 0.
    _max_pn_cd_epochs = max_pn_cd_epochs
    if epoch == 0:
        _max_pn_cd_epochs = min_pn_cd_epochs
        for idx, j in enumerate(ws):
            tmp = 0.
            for i in range(X_indptr[j], X_indptr[j+1]):
                tmp += X_data[i] * grad_datafit[X_indices[i]]
            old_grad[idx] = tmp
    else:
        pn_tol = pn_tol_ratio * pn_grad_diff

    n_performed_cd_epochs = 0
    for pn_cd_epoch in range(_max_pn_cd_epochs):
        weighted_fix_point_crit = 0.
        n_performed_cd_epochs += 1
        for idx, j in enumerate(ws):
            stepsize = 1/lc[idx] if lc[idx] != 0 else 1000
            old_value = w[j] + delta_w[idx]
            # TODO: Is it the correct gradient? What's the intuition behind the formula?
            tmp = 0.
            for i in range(X_indptr[j], X_indptr[j+1]):
                tmp += X_data[i] * X_delta_w[X_indices[i]] * hessian_diag[X_indices[i]]
            grad_j = old_grad[idx] + tmp

            new_value = penalty.prox_1d(
                old_value - grad_j * stepsize, stepsize, j)
            diff = new_value - old_value
            if diff != 0:
                weighted_fix_point_crit += (diff * lc[idx]) ** 2
                delta_w[idx] = new_value - w[j]
                for i in range(X_indptr[j], X_indptr[j+1]):
                    X_delta_w[X_indices[i]] += diff * X_data[i]
            # if max(verbose - 1, 0):
            #     print("delta w is ", delta_w[idx])
        # weighted_fix_point_crit /= X.shape[0] ** 2
        # TODO: beware scaling weighted_fix_point_crit and pn_tol
        # try a more proncipled criterion? subdiff dist?
        if weighted_fix_point_crit / len(hessian_diag) ** 2 <= pn_tol and pn_cd_epoch + 1 >= min_pn_cd_epochs:
            if max(verbose - 1, 0):
                print("Exited! weighted_fix_point_crit: ", weighted_fix_point_crit)
            break
    return delta_w, X_delta_w, n_performed_cd_epochs



@njit
def _backtrack_line_search(w, Xw, delta_w, X_delta_w, ws, y, penalty, max_backtrack):
    step_size = 1.
    prev_step_size = 0.
    for _ in range(max_backtrack):
        diff_step_size = step_size - prev_step_size
        delta_obj = 0.
        for idx, j in enumerate(ws):
            # w_j_old = w[j]
            w[j] += diff_step_size * delta_w[idx]
            # then TODO optimize code by creating a penalty.value_1D function
            # delta_obj += diff_step_size * (
            #     penalty.value(np.array([w[j]]))
            #     - penalty.value(np.array([w_j_old])))
            delta_obj += penalty.delta_pen(w[j], delta_w[idx])
        Xw += diff_step_size * X_delta_w
        grad = -y * sigmoid(-y * Xw)
        delta_obj += step_size * X_delta_w @ grad / \
            len(X_delta_w)  # TODO: might miss a step size
        if delta_obj < 1e-7:
            break
        prev_step_size = step_size
        step_size = step_size / 2
    if step_size != 1.0:
        X_delta_w *= step_size


@njit
def _compute_grad_hessian_datafit(X, y, Xw, ws, hessian_diag, grad_datafit, lc):
    n_samples = X.shape[0]
    for i in range(n_samples):
        prob = 1. / (1. + np.exp(y[i] * Xw[i]))
        # this supposes that the datafit is scaled by n_samples
        hessian_diag[i] = prob * (1. - prob) / n_samples
        grad_datafit[i] = -y[i] * prob / n_samples
    for idx, j in enumerate(ws):
        lc[idx] = ((X[:, j] ** 2) * hessian_diag).sum()


@njit
def _compute_grad_hessian_datafit_s(X_data, X_indptr, X_indices, 
                                    y, Xw, ws, hessian_diag, grad_datafit, lc):
    n_samples = len(y)
    for i in range(n_samples):
        prob = 1. / (1. + np.exp(y[i] * Xw[i]))
        # this supposes that the datafit is scaled by n_samples
        hessian_diag[i] = prob * (1. - prob) / n_samples
        grad_datafit[i] = -y[i] * prob / n_samples
    
    for idx, j in enumerate(ws):
        tmp = 0.
        for i in range(X_indptr[j], X_indptr[j+1]):
            tmp += hessian_diag[X_indices[i]] * X_data[i] ** 2
        lc[idx] = tmp
