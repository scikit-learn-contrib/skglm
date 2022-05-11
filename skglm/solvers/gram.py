import numpy as np
from numba import njit
from scipy import sparse

from skglm.datafits import Quadratic, Quadratic_32
from skglm.solvers.cd_utils import (
    construct_grad, construct_grad_sparse, dist_fix_point, prox_vec)


def cd_gram_quadratic(X, y, penalty, max_epochs=1000, tol=1e-4, w_init=None,
                      ws_strategy="subdiff", verbose=0):
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

    ws_strategy : ('subdiff'|'fixpoint'), optional
        The score used to compute the stopping criterion.

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
    G = X.T @ X  # gram matrix
    grads = (X.T @ y - G @ w_init) / len(y) if w_init is not None else X.T @ y / len(y)
    w = w_init.copy() if w_init is not None else np.zeros(n_features, dtype=X.dtype)
    for epoch in range(max_epochs):
        if is_sparse:
            _cd_epoch_gram_sparse(
                G.data, G.indptr, G.indices, grads, w, datafit, penalty, n_samples,
                n_features)
        else:
            _cd_epoch_gram(G, grads, w, datafit, penalty, n_samples, n_features)
        if epoch % 50 == 0:
            Xw = X @ w
            p_obj = datafit.value(y, w, Xw) + penalty.value(w)

            if is_sparse:
                grad = construct_grad_sparse(
                    X.data, X.indptr, X.indices, y, w, Xw, datafit, all_feats)
            else:
                grad = construct_grad(X, y, w, Xw, datafit, all_feats)
            if ws_strategy == "subdiff":
                opt_ws = penalty.subdiff_distance(w, grad, all_feats)
            elif ws_strategy == "fixpoint":
                opt_ws = dist_fix_point(w, grad, datafit, penalty, all_feats)

            stop_crit = np.max(opt_ws)
            if max(verbose - 1, 0):
                print(f"Epoch {epoch + 1}, objective {p_obj:.10f}, "
                      f"stopping crit {stop_crit:.2e}")
            if stop_crit <= tol:
                break
            obj_out.append(p_obj)
    return w, np.array(obj_out), stop_crit


@njit
def _cd_epoch_gram(G, grads, w, datafit, penalty, n_samples, n_features):
    """Run an epoch of coordinate descent in place with gradient update using Gram.

    Parameters
    ----------
    G : array, shape (n_features, n_features)
        Gram matrix.

    grads : array, shape (n_features,)
        Gradient vector.

    w : array, shape (n_features,)
        Coefficient vector.

    datafit : Datafit
        Datafit.

    penalty : Penalty
        Penalty.

    n_samples : int
        Number of samples.

    n_features : int
        Number of features.
    """
    lc = datafit.lipschitz
    for j in range(n_features):
        if lc[j] == 0:
            continue
        old_w_j = w[j]
        stepsize = 1 / lc[j] if lc[j] != 0 else 1000
        w[j] = penalty.prox_1d(old_w_j + grads[j] * stepsize, stepsize, j)
        if old_w_j != w[j]:
            grads += G[j, :] * (old_w_j - w[j]) / n_samples


@njit
def _cd_epoch_gram_sparse(G_data, G_indptr, G_indices, grads, w, datafit, penalty,
                          n_samples, n_features):
    """Run a CD epoch with Gram update for sparse design matrices.

    Parameters
    ----------
    G_data : array, shape (n_elements,)
        `data` attribute of the sparse CSC matrix G.

    G_indptr : array, shape (n_features + 1,)
        `indptr` attribute of the sparse CSC matrix G.

    G_indices : array, shape (n_elements,)
        `indices` attribute of the sparse CSC matrix G.

    grads : array, shape (n_features,)
        Gradient vector.

    w : array, shape (n_features,)
        Coefficient vector.

    datafit : Datafit
        Datafit.

    penalty : Penalty
        Penalty.

    n_samples : int
        Number of samples.

    n_features : int
        Number of features.
    """
    lc = datafit.lipschitz
    for j in range(n_features):
        if lc[j] == 0:
            continue
        old_w_j = w[j]
        stepsize = 1 / lc[j]
        w[j] = penalty.prox_1d(old_w_j + grads[j] / lc[j], stepsize, j)
        diff = old_w_j - w[j]
        if diff != 0:
            for i in range(G_indptr[j], G_indptr[j + 1]):
                grads[G_indices[i]] += diff * G_data[i] / n_samples


def fista_gram_quadratic(X, y, penalty, max_epochs=1000, tol=1e-4, w_init=None,
                         ws_strategy="subdiff", verbose=False):
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

    ws_strategy : ('subdiff'|'fixpoint'), optional
        The score used to compute the stopping criterion.

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
        w = prox_vec(penalty, z, 1/lc, n_features)
        z = w + (t_old - 1.) / t_new * (w - w_old)
        if epoch % 10 == 0:
            Xw = X @ w
            p_obj = datafit.value(y, w, Xw) + penalty.value(w)

            if is_sparse:
                grad = construct_grad_sparse(
                    X.data, X.indptr, X.indices, y, w, Xw, datafit, all_feats)
            else:
                grad = construct_grad(X, y, w, Xw, datafit, all_feats)
            if ws_strategy == "subdiff":
                opt_ws = penalty.subdiff_distance(w, grad, all_feats)
            elif ws_strategy == "fixpoint":
                opt_ws = dist_fix_point(w, grad, datafit, penalty, all_feats)

            stop_crit = np.max(opt_ws)
            if max(verbose - 1, 0):
                print(f"Epoch {epoch + 1}, objective {p_obj:.10f}, "
                      f"stopping crit {stop_crit:.2e}")
            if stop_crit <= tol:
                break
            obj_out.append(p_obj)
    return w, np.array(obj_out), stop_crit
