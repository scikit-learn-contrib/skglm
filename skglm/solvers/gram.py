import numpy as np
from numba import njit
from numpy.linalg import norm
from celer.homotopy import _grp_converter

from skglm.utils import BST, ST


@njit
def primal(alpha, y, X, w, weights):
    r = y - X @ w
    p_obj = (r @ r) / (2 * len(y))
    return p_obj + alpha * np.sum(np.abs(w * weights))


@njit
def primal_grp(alpha, y, X, w, grp_ptr, grp_indices, weights):
    r = y - X @ w
    p_obj = (r @ r) / (2 * len(y))
    for g in range(len(grp_ptr) - 1):
        w_g = w[grp_indices[grp_ptr[g]:grp_ptr[g + 1]]]
        p_obj += alpha * norm(w_g * weights[g], ord=2)
    return p_obj


def gram_lasso(X, y, alpha, max_iter, tol, w_init=None, weights=None, check_freq=10):
    p_obj_prev = np.inf
    n_features = X.shape[1]
    grads = X.T @ y / len(y)
    G = X.T @ X
    lipschitz = np.zeros(n_features, dtype=X.dtype)
    for j in range(n_features):
        lipschitz[j] = (X[:, j] ** 2).sum() / len(y)
    w = w_init.copy() if w_init is not None else np.zeros(n_features)
    z = w_init.copy() if w_init is not None else np.zeros(n_features)
    beta_0 = beta_1 = 1
    weights = weights if weights is not None else np.ones(n_features)
    # CD
    for n_iter in range(max_iter):
        beta_1 = (1 + np.sqrt(1 + 4 * beta_0 ** 2)) / 2
        cd_epoch(X, G, grads, w, z, alpha, beta_1, beta_0, lipschitz, weights)
        beta_0 = beta_1
        if n_iter % check_freq == 0:
            p_obj = primal(alpha, y, X, w, weights)
            if p_obj_prev - p_obj < tol:
                print("Convergence reached!")
                break
            print(f"iter {n_iter} :: p_obj {p_obj}")
            p_obj_prev = p_obj
    return w


def gram_group_lasso(X, y, alpha, groups, max_iter, tol, w_init=None, weights=None, 
                     check_freq=50):
    p_obj_prev = np.inf
    n_features = X.shape[1]
    grp_ptr, grp_indices = _grp_converter(groups, X.shape[1])
    n_groups = len(grp_ptr) - 1
    grads = X.T @ y / len(y)
    G = X.T @ X
    lipschitz = np.zeros(n_groups, dtype=X.dtype)
    for g in range(n_groups):
        X_g = X[:, grp_indices[grp_ptr[g]:grp_ptr[g + 1]]]
        lipschitz[g] = norm(X_g, ord=2) ** 2 / len(y)
    w = w_init.copy() if w_init is not None else np.zeros(n_features)
    weights = weights if weights is not None else np.ones(n_groups)
    # BCD
    for n_iter in range(max_iter):
        bcd_epoch(X, G, grads, w, alpha, lipschitz, grp_indices, grp_ptr, weights)
        if n_iter % check_freq == 0:
            p_obj = primal_grp(alpha, y, X, w, grp_ptr, grp_indices, weights)
            if p_obj_prev - p_obj < tol:
                print("Convergence reached!")
                break
            print(f"iter {n_iter} :: p_obj {p_obj}")
            p_obj_prev = p_obj
    return w


@njit
def cd_epoch(X, G, grads, w, z, alpha, beta_1, beta_0, lipschitz, weights):
    n_features = X.shape[1]
    for j in range(n_features):
        if lipschitz[j] == 0. or weights[j] == np.inf:
            continue
        old_w_j = w[j]
        old_z_j = z[j]
        w[j] = ST(z[j] + grads[j] / lipschitz[j], alpha / lipschitz[j] * weights[j])
        z[j] = w[j] + ((beta_0 - 1) / beta_1) * (w[j] - old_w_j)
        if old_z_j != z[j]:
            grads += G[j, :] * (old_z_j - z[j]) / len(X)


@njit
def bcd_epoch(X, G, grads, w, alpha, lipschitz, grp_indices, grp_ptr, weights):
    n_groups = len(grp_ptr) - 1
    for g in range(n_groups):
        if lipschitz[g] == 0. and weights[g] == np.inf:
            continue
        idx = grp_indices[grp_ptr[g]:grp_ptr[g + 1]]
        old_w_g = w[idx].copy()
        w[idx] = BST(w[idx] + grads[idx] / lipschitz[g], alpha / lipschitz[g]
                     * weights[g])
        diff = old_w_g - w[idx]
        if np.any(diff != 0.):
            grads += diff @ G[idx, :] / len(X)
