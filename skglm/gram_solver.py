from time import time
import numpy as np
from numpy.linalg import norm
from numba import njit
from celer import Lasso, GroupLasso
from benchopt.datasets.simulated import make_correlated_data
from skglm.utils import BST, ST


def _grp_converter(groups, n_features):
    if isinstance(groups, int):
        grp_size = groups
        if n_features % grp_size != 0:
            raise ValueError("n_features (%d) is not a multiple of the desired"
                             " group size (%d)" % (n_features, grp_size))
        n_groups = n_features // grp_size
        grp_ptr = grp_size * np.arange(n_groups + 1)
        grp_indices = np.arange(n_features)
    elif isinstance(groups, list) and isinstance(groups[0], int):
        grp_indices = np.arange(n_features).astype(np.int32)
        grp_ptr = np.cumsum(np.hstack([[0], groups]))
    elif isinstance(groups, list) and isinstance(groups[0], list):
        grp_sizes = np.array([len(ls) for ls in groups])
        grp_ptr = np.cumsum(np.hstack([[0], grp_sizes]))
        grp_indices = np.array([idx for grp in groups for idx in grp])
    else:
        raise ValueError("Unsupported group format.")
    return grp_ptr.astype(np.int32), grp_indices.astype(np.int32)


@njit
def primal(alpha, y, X, w):
    r = y - X @ w
    p_obj = (r @ r) / (2 * len(y))
    return p_obj + alpha * np.sum(np.abs(w))


@njit
def primal_grp(alpha, y, X, w, grp_ptr, grp_indices):
    r = y - X @ w
    p_obj = (r @ r) / (2 * len(y))
    for g in range(len(grp_ptr) - 1):
        w_g = w[grp_indices[grp_ptr[g]:grp_ptr[g + 1]]]
        p_obj += alpha * norm(w_g, ord=2)
    return p_obj


@njit
def cd_epoch(X, G, grads, w, alpha, lipschitz):
    n_features = X.shape[1]
    for j in range(n_features):
        if lipschitz[j] == 0.:
            continue
        old_w_j = w[j]
        w[j] = ST(w[j] + grads[j] / lipschitz[j], alpha / lipschitz[j])
        if old_w_j != w[j]:
            grads += G[j, :] * (old_w_j - w[j]) / len(X)


@njit
def bcd_epoch(X, G, grads, w, alpha, lipschitz, grp_indices, grp_ptr):
    n_groups = len(grp_ptr) - 1
    for g in range(n_groups):
        if lipschitz[g] == 0.:
            continue
        idx = grp_indices[grp_ptr[g]:grp_ptr[g + 1]]
        old_w_g = w[idx].copy()
        w[idx] = BST(w[idx] + grads[idx] / lipschitz[g], alpha / lipschitz[g])
        diff = old_w_g - w[idx]
        if np.any(diff != 0.):
            grads += diff @ G[idx, :] / len(X)


def lasso(X, y, alpha, max_iter, tol, check_freq=10):
    p_obj_prev = np.inf
    n_features = X.shape[1]
    # Initialization
    grads = X.T @ y / len(y)
    G = X.T @ X
    lipschitz = np.zeros(n_features, dtype=X.dtype)
    for j in range(n_features):
        lipschitz[j] = (X[:, j] ** 2).sum() / len(y)
    w = np.zeros(n_features)
    # CD
    for n_iter in range(max_iter):
        cd_epoch(X, G, grads, w, alpha, lipschitz)
        if n_iter % check_freq == 0:
            p_obj = primal(alpha, y, X, w)
            if p_obj_prev - p_obj < tol:
                print("Convergence reached!")
                break
            print(f"iter {n_iter} :: p_obj {p_obj}")
            p_obj_prev = p_obj
    return w


def group_lasso(X, y, alpha, groups, max_iter, tol, check_freq=50):
    p_obj_prev = np.inf
    n_features = X.shape[1]
    grp_ptr, grp_indices = _grp_converter(groups, X.shape[1])
    n_groups = len(grp_ptr) - 1
    # Initialization
    grads = X.T @ y / len(y)
    G = X.T @ X
    lipschitz = np.zeros(n_groups, dtype=X.dtype)
    for g in range(n_groups):
        X_g = X[:, grp_indices[grp_ptr[g]:grp_ptr[g + 1]]]
        lipschitz[g] = norm(X_g, ord=2) ** 2 / len(y)
    w = np.zeros(n_features)
    # BCD
    for n_iter in range(max_iter):
        bcd_epoch(X, G, grads, w, alpha, lipschitz, grp_indices, grp_ptr)
        if n_iter % check_freq == 0:
            p_obj = primal_grp(alpha, y, X, w, grp_ptr, grp_indices)
            if p_obj_prev - p_obj < tol:
                print("Convergence reached!")
                break
            print(f"iter {n_iter} :: p_obj {p_obj}")
            p_obj_prev = p_obj
    return w


if __name__ == "__main__":
    n_samples, n_features = 1_000_000, 300
    X, y, w_star = make_correlated_data(
        n_samples=n_samples, n_features=n_features, random_state=0)
    alpha_max = norm(X.T @ y, ord=np.inf)

    # Hyperparameters
    max_iter = 1000
    tol = 1e-8
    reg = 0.1
    group_size = 3

    alpha = alpha_max * reg / n_samples

    # Lasso
    print("#" * 15)
    print("Lasso")
    print("#" * 15)
    start = time()
    w = lasso(X, y, alpha, max_iter, tol)
    gram_lasso_time = time() - start
    clf_sk = Lasso(alpha, tol=tol, fit_intercept=False)
    start = time()
    clf_sk.fit(X, y)
    celer_lasso_time = time() - start
    np.testing.assert_allclose(w, clf_sk.coef_, rtol=1e-5)

    print("\n")
    print("Celer: %.2f" % celer_lasso_time)
    print("Gram: %.2f" % gram_lasso_time)
    print("\n")

    # Group Lasso
    print("#" * 15)
    print("Group Lasso")
    print("#" * 15)
    start = time()
    w = group_lasso(X, y, alpha, group_size, max_iter, tol)
    gram_group_lasso_time = time() - start
    clf_celer = GroupLasso(group_size, alpha, tol=tol, fit_intercept=False)
    start = time()
    clf_celer.fit(X, y)
    celer_group_lasso_time = time() - start
    np.testing.assert_allclose(w, clf_celer.coef_, rtol=1e-1)

    print("\n")
    print("Celer: %.2f" % celer_group_lasso_time)
    print("Gram: %.2f" % gram_group_lasso_time)
    print("\n")
