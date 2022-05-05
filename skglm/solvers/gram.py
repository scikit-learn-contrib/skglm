import numpy as np
from numba import njit
# from numpy.linalg import norm

from skglm.utils import BST, ST, ST_vec
from skglm.datafits import Quadratic


def cd_gram_quadratic(X, y, penalty, max_iter, tol, w_init=None, check_freq=100):
    """Gram solver for quadratic datafit."""
    n_features = X.shape[1]
    datafit = Quadratic()
    datafit.initialize(X, y)  # todo sparse
    G = X.T @ X  # gram matrix
    grads = X.T @ y / len(y)  # this is wrong if an init is used
    w = w_init.copy() if w_init is not None else np.zeros(n_features)
    for n_iter in range(max_iter):
        _cd_epoch_gram(X, G, grads, w, penalty, datafit)
        if n_iter % check_freq == 0:
            # check KKT
            # print(f"iter {n_iter} :: p_obj {p_obj:.5f} :: d_obj {d_obj:.5f}" +
            #       f" :: gap {d_gap:.5f}")
            # if d_gap < tol:
            #     print("Convergence reached!")
            #     break
    return w


def fista_gram_quadratic(
        X, y, penalty, max_iter, tol, w_init=None, check_freq=100):
    n_samples, n_features = X.shape
    norm_y2 = y @ y
    t_new = 1
    w = w_init.copy() if w_init is not None else np.zeros(n_features)
    z = w_init.copy() if w_init is not None else np.zeros(n_features)
    G = X.T @ X
    Xty = X.T @ y
    L = np.linalg.norm(X, ord=2) ** 2 / n_samples
    for n_iter in range(max_iter):
        t_old = t_new
        t_new = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2
        w_old = w.copy()
        z -= (G @ z - Xty) / L / n_samples
        w = ST_vec(z, alpha / L)  # use penalty.prox
        z = w + (t_old - 1.) / t_new * (w - w_old)
        if n_iter % check_freq == 0:
            pass
            # use KKT instead
            # print(f"iter {n_iter} :: p_obj {p_obj:.5f} :: d_obj {d_obj:.5f} " +
            #   f":: gap {d_gap:.5f}")
            # if d_gap < tol:
            # print("Convergence reached!")
            # break
    return w


@njit
def _cd_epoch_gram(X, G, grads, w, penalty, datafit):
    n_features = X.shape[1]
    for j in range(n_features):
        if lipschitz[j] == 0:
            continue
        old_w_j = w[j]
        # use penalty.prox1d
        w[j] = ST(w[j] + grads[j] / datafit.lipschitz[j],
                  alpha / lipschitz[j] * )
        if old_w_j != w[j]:
            grads += G[j, :] * (old_w_j - w[j]) / len(X)
