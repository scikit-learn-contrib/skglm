import numpy as np
from scipy import sparse
from numba import njit
from skglm.datafits import Logistic, Logistic_32
from skglm.solvers.common import construct_grad
from skglm.utils import AndersonAcceleration


def prox_newton_solver(
    X, y, datafit, penalty, w, Xw, max_iter=50, max_epochs=1000, max_backtrack=10,
    min_pn_cd_epochs=2, max_pn_cd_epochs=10, p0=10, tol=1e-4, verbose=0):
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
    n_features = X.shape[1]
    pen = penalty.is_penalized(n_features)
    unpen = ~pen
    n_unpen = unpen.sum()
    obj_out = []
    all_feats = np.arange(n_features)
    stop_crit = np.inf  # initialize for case n_iter=0

    accelerator = AndersonAcceleration(K=5)

    is_sparse = sparse.issparse(X)
    for t in range(max_iter):
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
        ws = np.argpartition(opt, -ws_size)[-ws_size:]

        if verbose:
            print(f"Iteration {t + 1}, {ws_size} feats in subpb.")

        # 2) run prox newton on smaller subproblem
        for epoch in range(max_epochs):
            _max_pn_cd_epochs = 1 if epoch == 0 else max_pn_cd_epochs
            pn_tol = 0 if epoch == 0 else tol
            # TODO: support sparse matrices
            _prox_newton_iter(
                X, Xw, w, y, penalty, ws, min_pn_cd_epochs, _max_pn_cd_epochs,
                max_backtrack, pn_tol)

            w_acc, Xw_acc = accelerator.extrapolate(w, Xw)
            p_obj = datafit.value(y, w, Xw) + penalty.value(w)
            p_obj_acc = datafit.value(y, w_acc, Xw_acc) + penalty.value(w_acc)

            if p_obj_acc < p_obj:
                w[:] = w_acc
                Xw[:] = Xw_acc
                p_obj = p_obj_acc

            if epoch % 5 == 0:  # check every 5 epochs, PN epochs are expensive
                grad = construct_grad(X, y, w, Xw, datafit, ws)
                opt_ws = penalty.subdiff_distance(w, grad, ws)
                stop_crit_in = np.max(opt_ws)
                if max(verbose - 1, 0):
                    print(f"Epoch {epoch + 1}, objective {p_obj:.10f}, "
                          f"stopping crit {stop_crit_in:.2e}")
                tol_in = tol if ws_size == n_features else 0.3 * stop_crit
                if stop_crit_in <= tol_in:
                    if max(verbose - 1, 0):
                        print("Early exit")
                    break
        obj_out.append(p_obj)
    return w, np.array(obj_out), stop_crit


@njit
def _prox_newton_iter(
    X, Xw, w, y, penalty, ws, min_cd_epochs, max_cd_epochs, max_backtrack, tol):
    n_samples, ws_size = X.shape[0], len(ws)

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
        lc[idx] = ((X[:, j] ** 2) * hessian_diag).sum()
        bias[idx] = X[:, j] @ grad_datafit

    delta_w, X_delta_w = _newton_cd(
        X, w, ws, hessian_diag, bias, lc, penalty, min_cd_epochs, max_cd_epochs, tol)
    step_size = _backtrack_line_search(
        w, Xw, delta_w, X_delta_w, ws, y, penalty, max_backtrack)

    for idx, j in enumerate(ws):
        w[j] += step_size * delta_w[idx]
    Xw += step_size * X_delta_w


@njit
def _newton_cd(
    X, w, feats, hessian_diag, bias, lc, penalty, min_epochs, max_epochs, tol):
    delta_w, X_delta_w = np.zeros(feats.shape[0]), np.zeros(X.shape[0])
    for epoch in range(max_epochs):
        sum_sq_hess_diff = 0
        for idx, j in enumerate(feats):
            stepsize = 1/lc[idx] if lc[idx] != 0 else 1000
            old_value = w[j] + delta_w[idx]
            tmp = (X[:, j] * X_delta_w * hessian_diag).sum()
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
def _backtrack_line_search(w, Xw, delta_w, X_delta_w, feats, y, penalty, max_backtrack):
    step_size = 1.
    for _ in range(max_backtrack):
        delta = 0.
        for idx, j in enumerate(feats):
            if w[j] + step_size * delta_w[idx] < 0:
                delta -= penalty.alpha * delta_w[idx]
            elif w[j] + step_size * delta_w[idx] > 0:
                delta += penalty.alpha * delta_w[idx]
            else:
                delta -= penalty.alpha * abs(delta_w[idx])
        theta = -y * sigmoid(-y * (Xw + step_size * X_delta_w))
        delta += X_delta_w @ theta
        if delta < 1e-7:
            break
        step_size = step_size / 2
    return step_size


@njit
def sigmoid(x):
    """Vectorwise sigmoid."""
    out = 1 / (1 + np.exp(-x))
    return out
