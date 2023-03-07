import numpy as np
from numpy.linalg import norm

from numba import njit
from skglm.utils.prox_funcs import ST

from graph_hashing.solvers.utils import compute_obj, validate_input


class CD:
    r"""Solve problem using a Proximal Gradient Descent algorithm.

    Objective to optimize::

        min_S sum_k (1/2) * ||S_k - H_k^T S H_k||_F^2 + lambda * ||S||_1


    Parameters
    ----------
    max_iter: int, default=100
        Maximum number of iteration. One iteration is a one pass on
        all features.

    tol : float, default=1e-6
        Stopping criterion for the optimization.

    check_freq: int, default=10
        Frequency according to which check the stopping criterion.

    verbose : bool or int
        Amount of verbosity.
    """

    def __init__(self, max_iter=100, tol=1e-6, check_freq=10, verbose=False):
        self.max_iter = max_iter
        self.tol = tol
        self.check_freq = check_freq
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

        # lipchitz[i, j] = sum_k (||row_H_k_i|| ||row_H_k_j||)**2
        lipchitz = squared_norms_H.T @ squared_norms_H

        for it in range(self.max_iter):
            # inplace update of S, grad
            _cd_pass(tensor_H_k, lmbd, S, residuals, lipchitz)

            if self.verbose:
                p_obj = compute_obj(S, tensor_H_k, tensor_S_k, lmbd)
                dist = _compute_dist_subdiff(tensor_H_k, lmbd, residuals, S)

                stop_crit = np.max(dist)

                print(f"Iteration {it:4}: {p_obj=:.8f} {stop_crit=:.4e}")

            # check convergence
            if it % self.check_freq == 0:
                dist = _compute_dist_subdiff(tensor_H_k, lmbd, residuals, S)

                stop_crit = np.max(dist)

                if stop_crit <= self.tol:
                    break

        return S, stop_crit


@njit(["f8(f8[:, :, :], f8[:, :, :], i4, i4)",
       "f8(f8[:, :, :], f8[:, :, :], i8, i8)"])
def _compute_grad(tensor_H_k, residuals, i, j):
    grad_ij = 0.

    for H_k, residual_k in zip(tensor_H_k, residuals):
        grad_ij -= H_k[i].T @ residual_k @ H_k[j]

    return grad_ij


@njit(["(f8[:, :, :], f8[:, :, :], f8, i4, i4)",
       "(f8[:, :, :], f8[:, :, :], f8, i8, i8)"])
def _update_residuals(tensor_H_k, residuals, delta_s_ij, i, j):
    # inplace update of residuals
    for k, H_k in enumerate(tensor_H_k):
        residuals[k] -= delta_s_ij * np.outer(H_k[i], H_k[j])


@njit("(f8[:, :, :], f8, f8[:, :], f8[:, :, :], f8[:, :])")
def _cd_pass(tensor_H_k, lmbd, S, residuals, lipchitz):
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
            if delta_s_ij != 0:
                _update_residuals(tensor_H_k, residuals, delta_s_ij, i, j)


@njit("f8(f8[:, :, :], f8, f8[:, :, :], f8[:, :])")
def _compute_dist_subdiff(tensor_H_k, lmbd, residuals, S):
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
