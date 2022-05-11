import numpy as np
from numba import njit
from scipy import sparse

from skglm.datafits import Quadratic, Quadratic_32
from skglm.solvers.cd_utils import (
    construct_grad, construct_grad_sparse, dist_fix_point, _prox_vec)


def cd_gram_quadratic(X, y, penalty, max_epochs=1000, tol=1e-4, w_init=None,
                      verbose=0):
    r"""Run a coordinate descent solver using Gram update for quadratic datafit.

    This solver should be used when n_samples >> n_features. It does not implement any
    working set strategy and iteratively updates the gradients (n_features,) instead of
    the prediction Xw (n_samples,).

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        Training data.

    y : array, shape (n_samples,)
        Target values.

    penalty : instance of Penalty class
        Penalty used in the model.

    max_epochs : int, optional
        Maximum number of CD epochs.

    tol : float, optional
        The tolerance for the optimization.

    w_init : array, shape (n_features,), optional
        Initial coefficient vector.

    verbose : bool or int, optional
        Amount of verbosity. 0/False is silent.

    Returns
    -------
    w : array, shape (n_features,)
        Coefficients.

    obj_out : array, shape (n_iter,)
        Objective value at every outer iteration.

    stop_crit : array, shape (n_alphas,)
        Value of stopping criterion at convergence along the path.
    """
    is_sparse = sparse.issparse(X)
    n_samples = len(y)
    n_features = len(X.indptr) - 1 if is_sparse else X.shape[1]
    all_feats = np.arange(n_features)
    obj_out = []
    datafit = Quadratic_32() if X.dtype == np.float32 else Quadratic()
    if is_sparse:
        datafit.initialize_sparse(X.data, X.indptr, X.indices, y)
    else:
        datafit.initialize(X, y)
    XtX = (X.T @ X).toarray() if is_sparse else X.T @ X  # gram matrix
    grad = ((datafit.Xty - XtX @ w_init) / len(y) if w_init is not None
            else datafit.Xty / len(y))
    w = w_init.copy() if w_init is not None else np.zeros(n_features, dtype=X.dtype)
    for epoch in range(max_epochs):
        _cd_epoch_gram(XtX, grad, w, datafit, penalty, n_samples, n_features)
        if epoch % 50 == 0:
            Xw = X @ w
            p_obj = datafit.value(y, w, Xw) + penalty.value(w)

            if is_sparse:
                grad = construct_grad_sparse(
                    X.data, X.indptr, X.indices, y, w, Xw, datafit, all_feats)
            else:
                grad = construct_grad(X, y, w, Xw, datafit, all_feats)
            # stop criterion: fixpoint
            opt = dist_fix_point(w, grad, datafit, penalty, all_feats)
            stop_crit = np.max(opt)
            if max(verbose - 1, 0):
                print(f"Epoch {epoch + 1}, objective {p_obj:.10f}, "
                      f"stopping crit {stop_crit:.2e}")
            if stop_crit <= tol:
                break
            obj_out.append(p_obj)
    return w, np.array(obj_out), stop_crit


@njit
def _cd_epoch_gram(XtX, grad, w, datafit, penalty, n_samples, n_features):
    lc = datafit.lipschitz
    for j in range(n_features):
        if lc[j] == 0:
            continue
        old_w_j = w[j]
        stepsize = 1 / lc[j] if lc[j] != 0 else 1000
        w[j] = penalty.prox_1d(old_w_j + grad[j] * stepsize, stepsize, j)
        if old_w_j != w[j]:
            grad += XtX[j, :] * (old_w_j - w[j]) / n_samples


def fista_gram_quadratic(X, y, penalty, max_epochs=1000, tol=1e-4, w_init=None,
                         verbose=False):
    r"""Run an accelerated proximal gradient descent for quadratic datafit.

    This solver should be used when n_samples >> n_features. It does not implement any
    working set strategy and iteratively updates the gradients (n_features,) instead of
    the residuals (n_samples,).

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        Training data.

    y : array, shape (n_samples,)
        Target values.

    penalty : instance of Penalty class
        Penalty used in the model.

    max_epochs : int, optional
        Maximum number of proximal steps.

    tol : float, optional
        The tolerance for the optimization.

    w_init : array, shape (n_features,), optional
        Initial coefficient vector.

    verbose : bool or int, optional
        Amount of verbosity. 0/False is silent.

    Returns
    -------
    w : array, shape (n_features,)
        Coefficients.

    obj_out : array, shape (n_iter,)
        Objective value at every outer iteration.

    stop_crit : array, shape (n_alphas,)
        Value of stopping criterion at convergence along the path.
    """
    is_sparse = sparse.issparse(X)
    n_samples = len(y)
    n_features = len(X.indptr) - 1 if is_sparse else X.shape[1]
    all_feats = np.arange(n_features)
    obj_out = []
    datafit = Quadratic_32() if X.dtype == np.float32 else Quadratic()
    if is_sparse:
        datafit.initialize_sparse(X.data, X.indptr, X.indices, y)
    else:
        datafit.initialize(X, y)
    t_new = 1
    w = w_init.copy() if w_init is not None else np.zeros(n_features, dtype=X.dtype)
    z = w_init.copy() if w_init is not None else np.zeros(n_features, dtype=X.dtype)
    G = X.T @ X
    lc = np.linalg.norm(X, ord=2) ** 2 / n_samples
    for epoch in range(max_epochs):
        t_old = t_new
        t_new = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2
        w_old = w.copy()
        z -= (G @ z - datafit.Xty) / lc / n_samples
        w = _prox_vec(penalty, z, 1/lc)
        z = w + (t_old - 1.) / t_new * (w - w_old)
        if epoch % 10 == 0:
            Xw = X @ w
            p_obj = datafit.value(y, w, Xw) + penalty.value(w)

            if is_sparse:
                grad = construct_grad_sparse(
                    X.data, X.indptr, X.indices, y, w, Xw, datafit, all_feats)
            else:
                grad = construct_grad(X, y, w, Xw, datafit, all_feats)

            opt = dist_fix_point(w, grad, datafit, penalty, all_feats)
            stop_crit = np.max(opt)
            if max(verbose - 1, 0):
                print(f"Epoch {epoch + 1}, objective {p_obj:.10f}, "
                      f"stopping crit {stop_crit:.2e}")
            if stop_crit <= tol:
                break
            obj_out.append(p_obj)
    return w, np.array(obj_out), stop_crit
