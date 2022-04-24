from re import L
import numpy as np
from numba import njit
from numpy.linalg import norm
from celer.homotopy import _grp_converter

from skglm.utils import BST, ST, ST_vec


@njit
def primal(alpha, r, w, weights):
    p_obj = (r @ r) / (2 * len(r))
    return p_obj + alpha * np.sum(np.abs(w * weights))

@njit
def dual(alpha, norm_y2, theta, y):
    d_obj = - np.sum((y / (alpha * len(y)) - theta) ** 2)
    d_obj *= 0.5 * alpha ** 2 * len(y)
    d_obj += norm_y2 / (2 * len(y))
    return d_obj

@njit
def dnorm_l1(theta, X, weights):
    n_features = X.shape[1]
    scal = 0.
    for j in range(n_features):
        Xj_theta = X[:, j] @ theta
        scal = max(scal, Xj_theta / weights[j])
    return scal

@njit
def create_dual_point(r, alpha, X, y, weights):
    theta = r / (alpha * len(y))
    scal = dnorm_l1(theta, X, weights)
    if scal > 1.:
        theta /= scal
    return theta

@njit
def dual_gap(alpha, norm_y2, y, X, w, weights):
    r = y - X @ w
    p_obj = primal(alpha, r, w, weights)
    theta = create_dual_point(r, alpha, X, y, weights)
    d_obj = dual(alpha, norm_y2, theta, y)
    return p_obj, d_obj, p_obj - d_obj


@njit
def primal_grp(alpha, y, X, w, grp_ptr, grp_indices, weights):
    r = y - X @ w
    p_obj = (r @ r) / (2 * len(y))
    for g in range(len(grp_ptr) - 1):
        w_g = w[grp_indices[grp_ptr[g]:grp_ptr[g + 1]]]
        p_obj += alpha * norm(w_g * weights[g], ord=2)
    return p_obj


@njit
def compute_lipschitz(X, y):
    n_features = X.shape[1]
    lipschitz = np.zeros(n_features, dtype=X.dtype)
    for j in range(n_features):
        lipschitz[j] = (X[:, j] ** 2).sum() / len(y)
    return lipschitz


def gram_lasso(X, y, alpha, max_iter, tol, w_init=None, weights=None, check_freq=10):
    n_features = X.shape[1]
    norm_y2 = y @ y 
    grads = X.T @ y / len(y)
    G = X.T @ X
    lipschitz = compute_lipschitz(X, y)
    w = w_init.copy() if w_init is not None else np.zeros(n_features)
    weights = weights if weights is not None else np.ones(n_features)
    # CD
    for n_iter in range(max_iter):
        cd_epoch(X, G, grads, w, alpha, lipschitz, weights)
        if n_iter % check_freq == 0:
            p_obj, d_obj, d_gap = dual_gap(alpha, norm_y2, y, X, w, weights)
            print(f"iter {n_iter} :: p_obj {p_obj:.5f} :: d_obj {d_obj:.5f}" +
                  f" :: gap {d_gap:.5f}")
            if d_gap < tol:
                print("Convergence reached!")
                break
    return w


def gram_fista_lasso(X, y, alpha, max_iter, tol, w_init=None, weights=None, 
                     check_freq=10):
    n_samples, n_features = X.shape
    norm_y2 = y @ y
    t_new = 1

    w = w_init.copy() if w_init is not None else np.zeros(n_features)
    z = w_init.copy() if w_init is not None else np.zeros(n_features)
    weights = weights if weights is not None else np.ones(n_features)

    G = X.T @ X
    Xty = X.T @ y
    L = np.linalg.norm(X, ord=2) ** 2 / n_samples

    for n_iter in range(max_iter):
        t_old = t_new
        t_new = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2
        w_old = w.copy()
        z -= (G @ z - Xty) / L / n_samples
        w = ST_vec(z, alpha / L * weights)
        z = w + (t_old - 1.) / t_new * (w - w_old)

        if n_iter % check_freq == 0:
            p_obj, d_obj, d_gap = dual_gap(alpha, norm_y2, y, X, w, weights)
            print(f"iter {n_iter} :: p_obj {p_obj:.5f} :: d_obj {d_obj:.5f} " +
                  f":: gap {d_gap:.5f}")
            if d_gap < tol:
                print("Convergence reached!")
                break
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
def cd_epoch(X, G, grads, w, alpha, lipschitz, weights):
    n_features = X.shape[1]
    for j in range(n_features):
        if lipschitz[j] == 0. or weights[j] == np.inf:
            continue
        old_w_j = w[j]
        w[j] = ST(w[j] + grads[j] / lipschitz[j], alpha / lipschitz[j] * weights[j])
        if old_w_j != w[j]:
            grads += G[j, :] * (old_w_j - w[j]) / len(X)


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
