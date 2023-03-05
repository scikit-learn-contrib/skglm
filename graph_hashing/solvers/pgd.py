import numpy as np
from numpy.linalg import norm

from numba import njit
from skglm.utils.prox_funcs import ST_vec

from graph_hashing.solvers.utils import compute_obj, compute_residual_k


class PGD:
    r"""Solve problem using a Proximal Gradient Descent algorithm.

    Objective to optimize::

        min_S sum_k (1/2) * ||S_k - H_k^T S H_k||_F^2 + lambda * ||S||_1


    Parameters
    ----------
    max_iter: int, default=100
        Maximum number of iteration.

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
        # TODO: validate parameters

        # cast as float
        tensor_H_k = tensor_H_k.astype(float)
        tensor_S_k = tensor_S_k.astype(float)

        # init vars
        n_nodes, _ = tensor_H_k[0].shape
        S = np.zeros((n_nodes, n_nodes))
        stop_crit = 0.

        # grad step
        lipchitz_cst = _compute_lipchitz_cst(tensor_H_k)

        # handle case all H_k are zeros
        if lipchitz_cst == 0.:
            return S, stop_crit

        step = 1 / lipchitz_cst

        # prox grad iteration
        for it in range(self.max_iter):

            # compute gradient
            grad = _compute_grad(S, tensor_H_k, tensor_S_k)

            # prox grad step
            S[:] = ST_vec(S - step * grad, step * lmbd)

            if self.verbose:
                p_obj = compute_obj(S, tensor_H_k, tensor_S_k, lmbd)
                stop_crit = _compute_dist_subdiff(S, grad, lmbd)

                print(f"Iteration {it}: {p_obj=:.4e} {stop_crit=:.4e}")

            # check convergence using fixed point distance
            if it % self.check_freq == 0:
                stop_crit = _compute_dist_subdiff(S, grad, lmbd)

                if stop_crit <= self.tol:
                    break

        return S, stop_crit


def _compute_lipchitz_cst(tensor_H_k):
    # Lipchitz is approximated using ``sum_k ||H_k||**4``
    # tensor_H_k has shape (n_H_k, n_nodes, n_sup_nodes)

    arr_lipchitz_H_k = norm(tensor_H_k, ord=2, axis=(1, 2))
    return (arr_lipchitz_H_k ** 4).sum()


def _compute_grad(S, tensor_H_k, tensor_S_k):
    grad = np.zeros_like(S)

    for k, H_k in enumerate(tensor_H_k):
        residual_k = compute_residual_k(S, tensor_H_k, tensor_S_k, k)
        grad -= H_k @ residual_k @ H_k.T

    return grad


@njit
def _compute_dist_subdiff(S, grad, lmbd):
    max_dist = 0.
    n_nodes = len(S)

    for i in range(n_nodes):
        for j in range(n_nodes):
            s_ij = S[i, j]
            grad_ij = grad[i, j]

            # compute distance
            if s_ij == 0.:
                dist_ij = max(0, abs(grad_ij) - lmbd)
            else:
                dist_ij = abs(grad_ij + np.sign(s_ij) * lmbd)

            # keep max
            max_dist = max(max_dist, dist_ij)

    return max_dist
