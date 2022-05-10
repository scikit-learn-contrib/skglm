import numpy as np
from numba import njit


@njit
def dist_fix_point(w, grad, datafit, penalty, ws):
    """Compute the violation of the fixed point iterate scheme.

    Parameters
    ----------
    w : array, shape (n_features,)
        Coefficient vector.

    grad : array, shape (n_features,)
        Gradient.

    datafit: instance of BaseDatafit
        Datafit.

    penalty: instance of BasePenalty
        Penalty.

    ws : array, shape (n_features,)
        The working set.

    Returns
    -------
    dist_fix_point : array, shape (n_features,)
        Violation score for every feature.
    """
    dist_fix_point = np.zeros(ws.shape[0])
    for idx, j in enumerate(ws):
        lcj = datafit.lipschitz[j]
        if lcj != 0:
            dist_fix_point[idx] = np.abs(
                w[j] - penalty.prox_1d(w[j] - grad[idx] / lcj, 1. / lcj, j))
    return dist_fix_point


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

    ws : array, shape (n_features,)
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

    ws : array, shape (n_features,)
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


@njit
def prox_vec(penalty, z, stepsize, n_features):
    """Apply the proximal operator iteratively to a vector of weight.
    
    Parameters
    ----------
    penalty : instance of Penalty
        Penalty.
    
    z : array, shape (n_features,)
        Coefficient vector.
    
    stepsize : float
        Step size.
    
    n_features : int
        Number of features.
    """
    w = np.zeros(n_features)
    for j in range(n_features):
        w[j] = penalty.prox_1d(z[j], stepsize, j)
    return w
