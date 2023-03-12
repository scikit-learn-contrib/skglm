import numpy as np
from numpy.linalg import norm

from numba import njit


def compute_obj(X, y, lmbd, w):
    return 0.5 * norm(y - X @ w) ** 2 + lmbd * norm(w, ord=1)


def eval_opt_crit(X, y, lmbd, w):
    grad = X.T @ (X @ w - y)
    opt_crit = _compute_dist_subdiff(w, grad, lmbd)

    return opt_crit


@njit("f8(f8[:], f8[:], f8)")
def _compute_dist_subdiff(w, grad, lmbd):
    max_dist = 0.

    for i in range(len(w)):
        grad_i = grad[i]
        w_i = w[i]

        if w[i] == 0.:
            dist = max(abs(grad_i) - lmbd, 0.)
        else:
            dist = abs(grad_i + np.sign(w_i) * lmbd)

        max_dist = max(max_dist, dist)

    return max_dist
