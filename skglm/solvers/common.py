import numpy as np
from numba import njit
from numpy.linalg import norm


@njit
def dist_fix_point_cd(w, grad_ws, lipschitz_ws, datafit, penalty, ws):
    """Compute the violation of the fixed point iterate scheme for CD.

    Parameters
    ----------
    w : array, shape (n_features,)
        Coefficient vector.

    grad_ws : array, shape (ws_size,)
        Gradient restricted to the working set.

    lipschitz_ws :  array, shape (len(ws),)
        Coordinatewise gradient Lipschitz constants, restricted to working set.

    datafit: instance of BaseDatafit
        Datafit.

    penalty: instance of BasePenalty
        Penalty.

    ws : array, shape (len(ws),)
        The working set.

    Returns
    -------
    dist : array, shape (n_features,)
        Violation score for every feature.
    """
    dist = np.zeros(ws.shape[0], dtype=w.dtype)

    for idx, j in enumerate(ws):
        if lipschitz_ws[idx] == 0.:
            continue

        step_j = 1 / lipschitz_ws[idx]
        dist[idx] = np.abs(
            w[j] - penalty.prox_1d(w[j] - step_j * grad_ws[idx], step_j, j)
        )
    return dist


@njit
def dist_fix_point_bcd(
        w, grad_ws, lipschitz_ws, datafit, penalty, ws):
    """Compute the violation of the fixed point iterate scheme for BCD.

    Parameters
    ----------
    w : array, shape (n_features,)
        Coefficient vector.

    grad_ws : array, shape (ws_size,)
        Gradient restricted to the working set.

    lipschitz_ws :  array, shape (len(ws),)
        Coordinatewise gradient Lipschitz constants, restricted to working set.

    datafit: instance of BaseDatafit
        Datafit.

    penalty: instance of BasePenalty
        Penalty.

    ws : array, shape (len(ws),)
        The working set.

    Returns
    -------
    dist : array, shape (n_groups,)
        Violation score for every group.

    Note:
        ----
        ``grad_ws`` is a stacked array of gradients ``[grad_ws_1, grad_ws_2, ...]``.
    """
    n_groups = len(penalty.grp_ptr) - 1
    dist = np.zeros(n_groups, dtype=w.dtype)

    grad_ptr = 0
    for idx, g in enumerate(ws):
        if lipschitz_ws[idx] == 0.:
            continue
        grp_g_indices = penalty.grp_indices[penalty.grp_ptr[g]: penalty.grp_ptr[g+1]]

        grad_g = grad_ws[grad_ptr: grad_ptr + len(grp_g_indices)]
        grad_ptr += len(grp_g_indices)

        step_g = 1 / lipschitz_ws[idx]
        w_g = w[grp_g_indices]
        dist[idx] = norm(
            w_g - penalty.prox_1group(w_g - grad_g * step_g, step_g, g)
        )
    return dist


@njit
def construct_grad(X, y, w, Xw, datafit, ws):
    """Compute the gradient of the datafit restricted to the working set.

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        Design matrix.

    y : array, shape (n_samples,)
        Target vector.

    w : array, shape (n_features,)
        Coefficient vector.

    Xw : array, shape (n_samples, )
        Model fit.

    datafit : Datafit
        Datafit.

    ws : array, shape (ws_size,)
        The working set.

    Returns
    -------
    grad : array, shape (ws_size, n_tasks)
        The gradient restricted to the working set.
    """
    grad = np.zeros(ws.shape[0])
    for idx, j in enumerate(ws):
        grad[idx] = datafit.gradient_scalar(X, y, w, Xw, j)
    return grad


@njit
def construct_grad_sparse(data, indptr, indices, y, w, Xw, datafit, ws):
    """Compute the gradient of the datafit restricted to the working set.

    Parameters
    ----------
    data : array-like
        Data array of the matrix in CSC format.

    indptr : array-like
        CSC format index point array.

    indices : array-like
        CSC format index array.

    y : array, shape (n_samples, )
        Target matrix.

    w : array, shape (n_features,)
        Coefficient matrix.

    Xw : array, shape (n_samples, )
        Model fit.

    datafit : Datafit
        Datafit.

    ws : array, shape (ws_size,)
        The working set.

    Returns
    -------
    grad : array, shape (ws_size, n_tasks)
        The gradient restricted to the working set.
    """
    grad = np.zeros(ws.shape[0])
    for idx, j in enumerate(ws):
        grad[idx] = datafit.gradient_scalar_sparse(
            data, indptr, indices, y, Xw, j)
    return grad
