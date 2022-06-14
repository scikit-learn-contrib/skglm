import numpy as np
from scipy import sparse
from numba import njit
from skglm.datafits import Logistic, Logistic_32
from skglm.solvers.common import construct_grad, construct_grad_sparse, dist_fix_point
from skglm.utils import (
    sigmoid, weighted_dot, weighted_dot_sparse, xj_dot_sparse)


def prox_newton_solver(
    X, y, datafit, penalty, w, Xw, max_iter=50, max_epochs=50_000, max_backtrack=10,
    min_pn_cd_epochs=2, max_pn_cd_epochs=10, p0=10, tol=1e-4, ws_strategy="subdiff",
    verbose=0
):
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

    ws_strategy : ('subdiff'|'fixpoint'), optional
        The score used to build the working set.

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
    if ws_strategy not in ("subdiff", "fixpoint"):
        raise ValueError(f"Unsupported value for ws_strategy: {ws_strategy}")
    if not isinstance(datafit, (Logistic, Logistic_32)):
        raise ValueError("Prox-Newton solver only supports Logistic datafits.")
    n_samples, n_features = X.shape
    if n_features >= 10_000:
        raise Warning(
            "Prox-Newton solver can be prohibitively slow for high-dimensional " +
            "problems. We recommend using `cd_solver` instead for faster convergence")
    pen = penalty.is_penalized(n_features)
    unpen = ~pen
    n_unpen = unpen.sum()
    obj_out = []
    all_feats = np.arange(n_features)
    stop_crit = np.inf  # initialize for case n_iter=0

    is_sparse = sparse.issparse(X)
    for t in range(max_iter):
        if is_sparse:
            grad = datafit.full_grad_sparse(
                X.data, X.indptr, X.indices, y, Xw)
        else:
            grad = construct_grad(X, y, w, Xw, datafit, all_feats)

        if ws_strategy == "subdiff":
            opt = penalty.subdiff_distance(w, grad, all_feats)
        elif ws_strategy == "fixpoint":
            opt = dist_fix_point(w, grad, datafit, penalty, all_feats)
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
        ws = np.argpartition(opt, -ws_size)[-ws_size:]

        if verbose:
            print(f"Iteration {t + 1}, {ws_size} feats in subpb.")

        # 2) run prox newton on smaller subproblem
        for epoch in range(max_epochs):
            _max_pn_cd_epochs = 1 if epoch == 0 else max_pn_cd_epochs
            pn_tol = tol  # TODO: needs adjustment
            if is_sparse:
                _prox_newton_iter_sparse(
                    X.data, X.indptr, X.indices, Xw, w, y, penalty, ws,
                    min_pn_cd_epochs, _max_pn_cd_epochs, max_backtrack, pn_tol)
            else:
                _prox_newton_iter(
                    X, Xw, w, y, penalty, ws, min_pn_cd_epochs, _max_pn_cd_epochs,
                    max_backtrack, pn_tol)

            if epoch % 10 == 0:
                p_obj = datafit.value(y, w[ws], Xw) + penalty.value(w)

                if is_sparse:
                    grad = construct_grad_sparse(
                        X.data, X.indptr, X.indices, y, w, Xw, datafit, ws)
                else:
                    grad = construct_grad(X, y, w, Xw, datafit, ws)
                if ws_strategy == "subdiff":
                    opt_ws = penalty.subdiff_distance(w, grad, ws)
                elif ws_strategy == "fixpoint":
                    opt_ws = dist_fix_point(w, grad, datafit, penalty, ws)

                stop_crit_in = np.max(opt_ws)
                if max(verbose - 1, 0):
                    print(f"Epoch {epoch + 1}, objective {p_obj:.10f}, "
                          f"stopping crit {stop_crit_in:.2e}")
                if ws_size == n_features:
                    if stop_crit_in <= tol:
                        break
                else:
                    if stop_crit_in < 0.3 * stop_crit:
                        if max(verbose - 1, 0):
                            print("Early exit")
                        break
        obj_out.append(p_obj)
    return w, np.array(obj_out), stop_crit


@njit
def _prox_newton_iter(
    X, Xw, w, y, penalty, ws, min_cd_epochs, max_cd_epochs, max_backtrack, tol
):
    n_samples, ws_size = X.shape[0], ws.shape[0]

    hessian_diag = np.zeros(n_samples)  # hessian = X^T D X, with D = diag(f_i'')
    grad_datafit = np.zeros(n_samples)  # gradient of F(Xw)

    lc = np.zeros(ws_size)  # weighted Lipschitz constants
    bias = np.zeros(ws_size)

    for i in range(n_samples):
        prob = 1. / (1. + np.exp(y[i] * Xw[i]))
        # this supposes that the datafit is scaled by n_samples
        hessian_diag[i] = prob * (1. - prob) / n_samples
        grad_datafit[i] = -y[i] * prob / n_samples

    for idx, j in enumerate(ws):
        lc[idx] = weighted_dot(X, Xw, hessian_diag, j, ignore_b=True)
        bias[idx] = X[:, j] @ grad_datafit

    delta_w, X_delta_w = _newton_cd(
        X, w, ws, hessian_diag, bias, lc, penalty, min_cd_epochs, max_cd_epochs, tol)
    step_size = _backtrack_line_search(
        w, Xw, delta_w, X_delta_w, ws, y, penalty, max_backtrack)

    for idx, j in enumerate(ws):
        w[j] += step_size * delta_w[idx]
    Xw += step_size * X_delta_w


@njit
def _prox_newton_iter_sparse(
    data, indptr, indices, Xw, w, y, penalty, ws, min_cd_epochs, max_cd_epochs,
    max_backtrack, tol
):
    n_samples, ws_size = len(y), ws.shape[0]

    hessian_diag = np.zeros(n_samples)  # hessian = X^T D X, with D = diag(f_i'')
    grad_datafit = np.zeros(n_samples)  # gradient of F(Xw)

    lc = np.zeros(ws_size)  # weighted Lipschitz constants
    bias = np.zeros(ws_size)

    for i in range(n_samples):
        prob = 1. / (1. + np.exp(y[i] * Xw[i]))
        # this supposes that the datafit is scaled by n_samples
        hessian_diag[i] = prob * (1. - prob) / n_samples
        grad_datafit[i] = -y[i] * prob / n_samples

    for idx, j in enumerate(ws):
        lc[idx] = weighted_dot_sparse(
            data, indptr, indices, Xw, hessian_diag, j, ignore_b=True)
        bias[idx] = xj_dot_sparse(data, indptr, indices, j, grad_datafit)

    delta_w, X_delta_w = _newton_cd_sparse(
        data, indptr, indices, w, ws, hessian_diag, bias, lc, penalty,
        min_cd_epochs, max_cd_epochs, tol)
    step_size = _backtrack_line_search(
        w, Xw, delta_w, X_delta_w, ws, y, penalty, max_backtrack)

    for idx, j in enumerate(ws):
        w[j] += step_size * delta_w[idx]
    Xw += step_size * X_delta_w


@njit
def _newton_cd(
    X, w, feats, hessian_diag, bias, lc, penalty, min_epochs, max_epochs, tol
):
    delta_w, X_delta_w = np.zeros(feats.shape[0]), np.zeros(X.shape[0])
    for epoch in range(max_epochs):
        sum_sq_hess_diff = 0
        for idx, j in enumerate(feats):
            stepsize = 1/lc[idx] if lc[idx] != 0 else 1000
            old_value = w[j] + delta_w[idx]
            tmp = weighted_dot(X, X_delta_w, hessian_diag, j, ignore_b=False)
            new_value = penalty.prox_1d(
                old_value - (bias[idx] + tmp) * stepsize, stepsize, j)
            diff = new_value - old_value
            if diff != 0:
                sum_sq_hess_diff += (diff * lc[idx]) ** 2
                delta_w[idx] = new_value - w[j]
                for i in range(X.shape[0]):
                    X_delta_w[i] += diff * X[i, j]
        if sum_sq_hess_diff <= tol and epoch + 1 >= min_epochs:
            break
    return delta_w, X_delta_w


@njit
def _newton_cd_sparse(
    data, indptr, indices, w, feats, hessian_diag, bias, lc, penalty, min_epochs,
    max_epochs, tol
):
    delta_w, X_delta_w = np.zeros(feats.shape[0]), np.zeros(hessian_diag.shape[0])
    for epoch in range(max_epochs):
        sum_sq_hess_diff = 0
        for idx, j in enumerate(feats):
            stepsize = 1/lc[idx] if lc[idx] != 0 else 1000
            old_value = w[j] + delta_w[idx]
            tmp = weighted_dot_sparse(
                data, indptr, indices, X_delta_w, hessian_diag, j)
            new_value = penalty.prox_1d(
                old_value - (bias[idx] + tmp) * stepsize, stepsize, j)
            diff = new_value - old_value
            if diff != 0:
                sum_sq_hess_diff += (diff * lc[idx]) ** 2
                delta_w[idx] = new_value - w[j]
                for i in range(indptr[j], indptr[j + 1]):
                    X_delta_w[indices[i]] += diff * data[i]
        if sum_sq_hess_diff <= tol and epoch + 1 >= min_epochs:
            break
    return delta_w, X_delta_w


@njit
def _backtrack_line_search(w, Xw, delta_w, X_delta_w, ws, y, penalty, max_backtrack):
    step_size = 1.
    for _ in range(max_backtrack):
        delta = 0.
        for j in ws:
            if w[j] + step_size * delta_w[j] < 0:
                delta -= penalty.alpha * delta_w[j]
            elif w[j] + step_size * delta_w[j] > 0:
                delta += penalty.alpha * delta_w[j]
            else:
                delta -= penalty.alpha * abs(delta_w[j])
        theta = -y * sigmoid(-y * (Xw + step_size * X_delta_w))
        delta += X_delta_w @ theta
        if delta < 1e-7:
            break
        step_size = step_size / 2
    return step_size
