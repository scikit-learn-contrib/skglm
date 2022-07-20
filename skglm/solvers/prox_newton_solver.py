import numpy as np
from scipy import sparse
from numba import njit
# import random
from skglm.datafits import Logistic, Logistic_32
from skglm.datafits.single_task import sigmoid
from skglm.solvers.common import construct_grad


def prox_newton_solver(
        X, y, datafit, penalty, w, Xw, max_iter=50, max_epochs=1000, max_backtrack=20,
        min_pn_cd_epochs=2, max_pn_cd_epochs=20, p0=10, tol=1e-4, verbose=0,
        eps_in=0.3):
    r"""Run a prox-Newton solver.

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        Training data.

    y : array, shape (n_samples,)
        Target values.

    datafit : instance of Datafit class
        Datafitting term.

    penalty : instance of Penalty class
        Penalty used in the model.

    w : array, shape (n_features,)
        Coefficient vector.

    Xw : array, shape (n_samples,)
        Model fit.

    max_iter : int, optional
        Maximum number of iterations (definition of working set and
        resolution of problem restricted to features in working set).

    max_epochs : int, optional
        Maximum number of prox-Newton epochs on each subproblem.

    max_backtrack : int, optinal
        Maximum number of backtracking steps for the line search.

    min_pn_cd_epochs : int, optional
        Minimum number of iterations used in the prox-Newton coordinate
        descent.

    max_pn_cd_epochs : int, optional
        Maximum number of iterations used in the prox-Newton coordinate
        descent.

    p0 : int, optional
        First working set size.

    tol : float, optional
        The tolerance for the optimization.

    verbose : bool or int, optional
        Amount of verbosity. 0/False is silent.

    Returns
    -------
    w : array, shape (n_features,)
        Coefficient vector.

    obj_out : array, shape (n_iter,)
        The objective values at every outer iteration.

    stop_crit : float
        Value of stopping criterion at convergence.
    """
    if not isinstance(datafit, (Logistic, Logistic_32)):
        raise ValueError("Prox-Newton solver only supports Logistic datafits.")
    n_samples, n_features = X.shape
    pen = penalty.is_penalized(n_features)
    unpen = ~pen
    n_unpen = unpen.sum()
    obj_out = []
    all_feats = np.arange(n_features)
    stop_crit = np.inf  # initialize for case n_iter=0

    hessian_diag = np.zeros(n_samples)  # hessian = X^T D X, with D = diag(f_i'')
    grad_datafit = np.zeros(n_samples)  # gradient of F(Xw)

    is_sparse = sparse.issparse(X)
    for t in range(max_iter):
        if max(verbose - 1, 0):
            print("################################")
            print("Number iter outer", t)
        if is_sparse:
            grad = datafit.full_grad_sparse(
                X.data, X.indptr, X.indices, y, Xw)
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
        bias = np.zeros(ws_size)
        pn_grad_diff = 0.
        pn_tol_ratio = 10.

        _compute_grad_hessian_datafit(X, y, Xw, ws, hessian_diag, grad_datafit, lc)

        # 2) run prox newton on smaller subproblem
        for epoch in range(max_epochs):
            # TODO: support sparse matrices
            if max(verbose - 1, 0):
                print("################################")
                print("Number iter PN", epoch)
            pn_grad_diff = _prox_newton_iter(
                X, Xw, w, y, penalty, ws, min_pn_cd_epochs, max_pn_cd_epochs,
                max_backtrack, pn_tol_ratio, pn_grad_diff, hessian_diag, grad_datafit,
                lc, bias, epoch, verbose=verbose)

            p_obj = datafit.value(y, w, Xw) + penalty.value(w)

            grad = construct_grad(X, y, w, Xw, datafit, ws)
            opt_ws = penalty.subdiff_distance(w, grad, ws)
            stop_crit_in = np.max(opt_ws)
            if max(verbose - 1, 0):
                print(f"Epoch {epoch + 1}, objective {p_obj:.10f}, "
                      f"stopping crit {stop_crit_in:.2e}")
            tol_in = eps_in * stop_crit
            if stop_crit_in <= tol_in:
                if max(verbose - 1, 0):
                    print("Early exit")
                break
        obj_out.append(p_obj)
    return w, np.array(obj_out), stop_crit


@njit
def _prox_newton_iter(
        X, Xw, w, y, penalty, ws, min_pn_cd_epochs, max_pn_cd_epochs, max_backtrack,
        pn_tol_ratio, pn_grad_diff, hessian_diag, grad_datafit, lc, bias, epoch,
        verbose=0):

    delta_w, X_delta_w = _compute_descent_direction(
        X, w, ws, hessian_diag, bias, grad_datafit, lc, penalty, min_pn_cd_epochs, 
        max_pn_cd_epochs, pn_tol_ratio, pn_grad_diff, epoch, verbose=verbose)
    _backtrack_line_search(w, Xw, delta_w, X_delta_w, ws, y, penalty, max_backtrack)

    _compute_grad_hessian_datafit(X, y, Xw, ws, hessian_diag, grad_datafit, lc)

    pn_grad_diff = 0.
    for idx, j in enumerate(ws):
        actual_grad = X[:, j] @ grad_datafit
        approx_grad = bias[idx] + (X[:, j] * X_delta_w * hessian_diag).sum()

        bias[idx] = actual_grad
        grad_diff = actual_grad - approx_grad
        pn_grad_diff += grad_diff ** 2

    return pn_grad_diff


@njit
def _compute_descent_direction(
        X, w, ws, hessian_diag, bias, grad_datafit, lc, penalty, min_pn_cd_epochs, 
        max_pn_cd_epochs, pn_tol_ratio, pn_grad_diff, epoch, verbose=0):
    delta_w, X_delta_w = np.zeros(len(ws)), np.zeros(X.shape[0])
    pn_tol = 0.
    _max_pn_cd_epochs = max_pn_cd_epochs
    if epoch == 0:
        _max_pn_cd_epochs = min_pn_cd_epochs
        for idx, j in enumerate(ws):
            bias[idx] = X[:, j] @ grad_datafit
    else:
        pn_tol = pn_tol_ratio * pn_grad_diff

    if max(verbose - 1, 0):
        print("############################")
    for pn_cd_epoch in range(_max_pn_cd_epochs):
        sum_sq_hess_diff = 0.
        if max(verbose - 1, 0):
            print("Number iter cd ", pn_cd_epoch)
            print("ws size: ", len(ws))
        # random.shuffle(ws)
        for idx, j in enumerate(ws):
            stepsize = 1/lc[idx] if lc[idx] != 0 else 1000
            old_value = w[j] + delta_w[idx]
            # TODO: Is it the correct gradient? What's the intuition behind the formula? 
            grad_j = bias[idx] + (X[:, j] * X_delta_w * hessian_diag).sum()
            new_value = penalty.prox_1d(
                old_value - grad_j * stepsize, stepsize, j)
            diff = new_value - old_value
            if diff != 0:
                sum_sq_hess_diff += (diff * lc[idx]) ** 2
                delta_w[idx] = new_value - w[j]
                for i in range(X.shape[0]):
                    X_delta_w[i] += diff * X[i, j]
            # if max(verbose - 1, 0):
            #     print("delta w is ", delta_w[idx])
        # sum_sq_hess_diff /= X.shape[0] ** 2
        # TODO: beware scaling sum_sq_hess_diff and pn_tol
        if sum_sq_hess_diff <= pn_tol and pn_cd_epoch + 1 >= min_pn_cd_epochs:
            print("Exited! sum_sq_hess_diff: ", sum_sq_hess_diff)
            break
    return delta_w, X_delta_w


@njit
def _backtrack_line_search(w, Xw, delta_w, X_delta_w, ws, y, penalty, max_backtrack):
    step_size = 1.
    prev_step_size = 0.
    for _ in range(max_backtrack):
        diff_step_size = step_size - prev_step_size
        delta_obj = 0.
        for idx, j in enumerate(ws):
            w_j_old = w[j]
            w[j] += diff_step_size * delta_w[idx]
            # then TODO optimize code by creating a penalty.value_1D function
            # delta_obj += diff_step_size * (
            #     penalty.value(np.array([w[j]]))
            #     - penalty.value(np.array([w_j_old])))
            delta_obj += penalty.delta_pen(w[j], delta_w[idx])
        Xw += diff_step_size * X_delta_w
        grad = -y * sigmoid(-y * Xw)
        delta_obj += step_size * X_delta_w @ grad / len(X_delta_w)  # TODO: might miss a step size
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
