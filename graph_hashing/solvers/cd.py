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
        all_idx = np.arange(n_nodes ** 2)
        stop_crit = 0.

        # grad where grad[i, j] is grad w.r.t (i, j)
        grad = _init_grad(tensor_H_k, tensor_S_k)

        # squared norm of each row of H_k
        # squared_norms_H[k, i] is squared norm of the i-th row of H_k
        squared_norms_H = norm(tensor_H_k, axis=2) ** 2

        # prod_squared_norms_H[i, j] = sum_k (||row_H_k_i|| ||row_H_k_i||)**2
        prod_squared_norms_H = squared_norms_H.T @ squared_norms_H

        for it in range(self.max_iter):
            # inplace update of S, grad
            _cd_pass(S, grad, lmbd, prod_squared_norms_H, all_idx)

            if self.verbose:
                p_obj = compute_obj(S, tensor_H_k, tensor_S_k, lmbd)
                dist = _compute_dist_subdiff(S, grad, lmbd, all_idx)

                stop_crit = np.max(dist)

                print(f"Iteration {it}: {p_obj=:.4e} {stop_crit=:.4e}")

        return S, stop_crit


# specify signature in jit to avoid compilation overhead
# @jit(["UniTuple(i4, 2)(i4, f8[:, :])",
#       "UniTuple(i8, 2)(i8, f8[:, :])"],
#      nopython=True)
def _from_vec_idx_to_ij(vec_idx, S):
    # map idx in flatten representation to idx in matrix representation
    n_nodes = len(S)
    i, j = divmod(vec_idx, n_nodes)

    return i, j


# @jit(["(f8[:, :], f8[:, :], f8, f8[:, :], i4[:])",
#       "(f8[:, :], f8[:, :], f8, f8[:, :], i8[:])"],
#      nopython=True)
def _cd_pass(S, grad, lmbd, prod_squared_norms_H, ws):
    # inplace update of S, grad

    for vec_idx in ws:
        i, j = _from_vec_idx_to_ij(vec_idx, S)

        # skip zero coordinates
        if prod_squared_norms_H[i, j] == 0.:
            continue

        step = 1 / prod_squared_norms_H[i, j]

        old_s_ij = S[i, j]
        S[i, j] = ST(old_s_ij - step * grad[i, j], step * lmbd)

        # update grad
        grad[i, j] += (S[i, j] - old_s_ij) * prod_squared_norms_H[i, j]


# @jit("f8[:, :](f8[:, :, :], f8[:, :, :])",
#      nopython=True)
def _init_grad(tensor_H_k, tensor_S_k):
    n_H_k, n_nodes, _ = tensor_H_k.shape

    grad = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):

            for H_k, S_k in zip(tensor_H_k, tensor_S_k):
                grad[i, j] -= (S_k @ H_k[j]) @ H_k[i]

    return grad


# @jit(["f8[:](f8[:, :], f8[:, :], f8, i4[:])",
#       "f8[:](f8[:, :], f8[:, :], f8, i8[:])"],
#      nopython=True)
def _compute_dist_subdiff(S, grad, lmbd, ws):
    dist = np.zeros(len(ws))

    for vec_idx in ws:
        i, j = _from_vec_idx_to_ij(vec_idx, S)

        s_ij = S[i, j]
        grad_ij = grad[i, j]

        # compute distance
        if s_ij == 0.:
            dist_ij = max(0, abs(grad_ij) - lmbd)
        else:
            dist_ij = abs(grad_ij + np.sign(s_ij) * lmbd)

        dist[vec_idx] = dist_ij

    return dist
