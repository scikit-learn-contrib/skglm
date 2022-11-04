"""Solve an un-normalized Lasso problem using P-D method

Problem reads::

    min_x 1/2 * ||b - Ax||^2 + alpha ||x||_1

Which can be recast as a saddle point problem::

    max_y min_x <Ax, y> + g(x) - h*(y)

where::

    g(x) = alpha ||x||_1
    h(z) = (1/2) * ||b - z||^2   <===> h*(y) = <y, b> + (1/2) * ||y||^2
"""

import numpy as np
from numpy.linalg import norm

from skglm.utils import ST, ST_vec


def fb_lasso(A, b, alpha, max_iter=1000, verbose=0):
    # solves using Fercoq & Bianchi
    n_samples, n_features = A.shape

    sigma = 1 / (2 * n_features * norm(A, ord=2))
    tau = 1 / norm(A, axis=0, ord=2)

    all_features = np.arange(n_features)
    p_obj_out = []

    # primal variables
    x = np.zeros(n_features)

    # dual variables
    y = np.zeros(n_samples)
    y_bar = np.zeros(n_samples)

    for iter in range(max_iter):

        for j in all_features:
            # primal update
            old_x_j = x[j]
            x[j] = _prox_g_j(old_x_j - tau[j] * (A[:, j] @ (2 * y_bar - y)), tau[j],
                             alpha, b, A)

            # dual update
            y_bar = _prox_h_star(y + sigma * (A @ x), sigma,
                                 b, A)
            y += (y_bar - y) / n_features

            if verbose:
                print(f"Iter {iter+1}: {_compute_obj(b, A, x, alpha):.10f}")

        p_obj_out.append(_compute_obj(b, A, x, alpha))

    return x, p_obj_out


def cp_lasso(A, b, alpha, max_iter=1000, verbose=0):
    # using Chambolle Pock
    n_samples, n_features = A.shape

    L = norm(A, ord=2)
    sigma = 1 / L
    tau = 1 / L

    p_obj_out = []

    # primal var
    x = np.zeros(n_features)
    x_bar = np.zeros(n_features)

    # dual var
    y = np.zeros(n_samples)

    for iter in range(max_iter):

        # dual update
        y = _prox_h_star(y + sigma * (A @ x_bar), sigma, b, A)

        # primal update
        old_x = x.copy()
        x = _prox_g(x - tau * (A.T @ y), tau, alpha, b, A)
        x_bar = 2 * x - old_x

        if verbose:
            print(f"Iter {iter+1}: {_compute_obj(b, A, x, alpha):.10f}")

        p_obj_out.append(_compute_obj(b, A, x, alpha))

    return x, p_obj_out


def _compute_obj(b, A, x, alpha):
    return ((1/2) * norm(b - A @ x, ord=2) ** 2
            + alpha * norm(x, ord=1))


def _prox_g_j(x_j, step, alpha, b, A):
    return ST(x_j, step * alpha)


def _prox_g(x, step, alpha, b, A):
    return ST_vec(x, step * alpha)


def _prox_h_star(y, step, b, A):
    return (y/step - b) / (1 + 1/step)
