from itertools import product
import numpy as np
from numpy.linalg import norm

from numba import jit
from skglm.utils.prox_funcs import ST

from graph_hashing.solvers.utils import compute_obj, validate_input


class CD:
    r"""Solve problem using a Proximal Gradient Descent algorithm.

    Objective to optimize::

        min_S sum_k (1/2) * ||S_k - H_k^T S H_k||_F^2 + lambda * ||S||_1


    Parameters
    ----------
    max_iter: int, default=100
        Maximum number of iteration.

    verbose : bool or int
        Amount of verbosity.
    """

    def __init__(self, max_iter=100, verbose=False):
        self.max_iter = max_iter
        self.verbose = verbose

    def solve(self, tensor_H_k, tensor_S_k, lmbd):
        tensor_H_k, tensor_S_k = validate_input(tensor_H_k, tensor_S_k)

        # init vars
        n_nodes, _ = tensor_H_k[0].shape
        S = np.zeros((n_nodes, n_nodes))
        residuals = tensor_S_k.copy()
        stop_crit = 0.

        # squared norm of each row of H_k
        # squared_norms_H[k, i] is squared norm of the i-th row of H_k
        squared_norms_H = norm(tensor_H_k, axis=2) ** 2

        # lipchitz[i, j] = sum_k (||row_H_k_i|| ||row_H_k_i||)**2
        lipchitz = squared_norms_H.T @ squared_norms_H

        for it in range(self.max_iter):
            # inplace update of S, grad
            _cd_pass(tensor_H_k, S, residuals, lmbd, lipchitz)

            if self.verbose:
                p_obj = compute_obj(S, tensor_H_k, tensor_S_k, lmbd)
                dist = _compute_dist_subdiff(tensor_H_k, residuals, S, lmbd)

                stop_crit = np.max(dist)

                print(f"Iteration {it:4}: {p_obj=:.8f} {stop_crit=:.4e}")

        return S, stop_crit


def _cd_pass(tensor_H_k, S, residuals, lmbd, lipchitz):
    # inplace update of S, grad

    n_nodes = len(S)

    for i in range(n_nodes):
        for j in range(n_nodes):

            # skip zero coordinates
            if lipchitz[i, j] == 0.:
                continue

            step = 1 / lipchitz[i, j]

            grad_ij = _compute_grad(tensor_H_k, residuals, i, j)

            old_s_ij = S[i, j]
            S[i, j] = ST(old_s_ij - step * grad_ij, step * lmbd)

            # update grad
            delta_s_ij = S[i, j] - old_s_ij
            _update_residuals(tensor_H_k, residuals, delta_s_ij, i, j)


def _compute_grad(tensor_H_k, residuals, i, j):
    grad_ij = 0.

    for H_k, residual_k in zip(tensor_H_k, residuals):
        grad_ij -= (residual_k @ H_k[j]) @ H_k[i]

    return grad_ij


def _update_residuals(tensor_H_k, residuals, delta_s_ij, i, j):

    for k, H_k in enumerate(tensor_H_k):
        residuals[k] -= delta_s_ij * np.outer(H_k[i], H_k[j])


def _compute_dist_subdiff(tensor_H_k, residuals, S, lmbd):
    n_nodes = len(S)
    max_dist = 0.

    for i in range(n_nodes):
        for j in range(n_nodes):

            s_ij = S[i, j]
            grad_ij = _compute_grad(tensor_H_k, residuals, i, j)

            # compute distance
            if s_ij == 0.:
                dist_ij = max(0, abs(grad_ij) - lmbd)
            else:
                dist_ij = abs(grad_ij + np.sign(s_ij) * lmbd)

            max_dist = max(max_dist, dist_ij)

    return max_dist
