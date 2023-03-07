import numpy as np
from numpy.linalg import norm

from numba import njit
from skglm.utils.prox_funcs import ST

from graph_hashing.solvers.utils import compute_obj, validate_input

EPS_DROP = 0.3
FREQ_CHECK = 10


class CD_WS:
    r"""Solve problem using a CD algorithm with working sets.

    Working sets are built based on a distance to subdifferential.

    Objective to optimize::

        min_S sum_k (1/2) * ||S_k - H_k^T S H_k||_F^2 + lambda * ||S||_1


    Parameters
    ----------
    max_iter: int, default=100
        Maximum number of iteration. One iteration corresponds
        to solving one subproblem.

    max_epochs: int, default=1000
        Maximum number of epochs to perform. One iteration is a one pass
        over the coordinate of the working set.

    tol : float, default=1e-6
        Stopping criterion for the optimization.

    p0: int, default=10
        Number of coordinates to be include initially in the working set.

    verbose : bool or int
        Amount of verbosity.
    """

    def __init__(self, max_iter=100, max_epochs=1000, tol=1e-6, p0=100, verbose=False):
        self.max_iter = max_iter
        self.max_epochs = max_epochs
        self.tol = tol
        self.p0 = p0
        self.verbose = verbose

    def solve(self, tensor_H_k, tensor_S_k, lmbd):
        tensor_H_k, tensor_S_k = validate_input(tensor_H_k, tensor_S_k)

        # init vars
        n_nodes, _ = tensor_H_k[0].shape
        S = np.zeros((n_nodes, n_nodes))
        residuals = tensor_S_k.copy()
        stop_crit = 0.
        ws_size = self.p0

        all_coordinates = np.arange(n_nodes ** 2)

        # squared norm of each row of H_k
        # squared_norms_H[k, i] is squared norm of the i-th row of H_k
        squared_norms_H = norm(tensor_H_k, axis=2) ** 2

        # lipchitz[i, j] = sum_k (||row_H_k_i|| ||row_H_k_j||)**2
        lipchitz = squared_norms_H.T @ squared_norms_H

        for it in range(self.max_iter):

            # compute scores
            scores = _compute_dist_subdiff(
                tensor_H_k, lmbd, residuals, S, all_coordinates)

            # check convergence
            stop_crit = np.max(scores)

            if self.verbose:
                p_obj = compute_obj(S, tensor_H_k, tensor_S_k, lmbd)
                print(f"Iteration {it:4}: {p_obj=:.8f} {stop_crit=:.4e} {ws_size=}")

            if stop_crit <= self.tol:
                break

            # build working sets
            gsupp_size = (S != 0).sum()
            ws_size = min(
                max(int(1.5 * gsupp_size), self.p0),
                n_nodes**2
            )

            # k-largest items (no sort)
            ws = np.argpartition(scores, -ws_size)[-ws_size:]

            # solve subproblem on ws
            tol_in = EPS_DROP * stop_crit
            _inner_solver(tensor_H_k, lmbd, S, residuals,
                          lipchitz, ws, self.max_epochs, tol_in)

        return S, stop_crit


@njit(["UniTuple(i4, 2)(i4, f8[:, :])",
       "UniTuple(i8, 2)(i8, f8[:, :])"])
def _i_j_from_idx_coordinate(idx_coordinate, S):
    # return (i, j) idx position in S from idx in flatten S
    n_nodes = len(S)
    i, j = divmod(idx_coordinate, n_nodes)

    return i, j


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


@njit(["(f8[:, :, :], f8, f8[:, :], f8[:, :, :], f8[:, :], i4[:])",
       "(f8[:, :, :], f8, f8[:, :], f8[:, :, :], f8[:, :], i8[:])"])
def _cd_pass(tensor_H_k, lmbd, S, residuals, lipchitz, ws):
    # inplace update of S, grad

    for idx_coordinate in ws:
        i, j = _i_j_from_idx_coordinate(idx_coordinate, S)

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


@njit(["f8[:](f8[:, :, :], f8, f8[:, :, :], f8[:, :], i4[:])",
       "f8[:](f8[:, :, :], f8, f8[:, :, :], f8[:, :], i8[:])"])
def _compute_dist_subdiff(tensor_H_k, lmbd, residuals, S, ws):
    dist = np.zeros(len(ws))

    for idx, idx_coordinate in enumerate(ws):
        i, j = _i_j_from_idx_coordinate(idx_coordinate, S)

        s_ij = S[i, j]
        grad_ij = _compute_grad(tensor_H_k, residuals, i, j)

        # compute distance
        if s_ij == 0.:
            dist_ij = max(0, abs(grad_ij) - lmbd)
        else:
            dist_ij = abs(grad_ij + np.sign(s_ij) * lmbd)

        dist[idx] = dist_ij

    return dist


@njit(["(f8[:, :, :], f8, f8[:, :], f8[:, :, :], f8[:, :], i4[:], i4, f8)",
       "(f8[:, :, :], f8, f8[:, :], f8[:, :, :], f8[:, :], i8[:], i4, f8)"])
def _inner_solver(tensor_H_k, lmbd,
                  S, residuals, lipchitz, ws,
                  n_epochs, tol_in):

    for epoch in range(n_epochs):
        _cd_pass(tensor_H_k, lmbd, S, residuals, lipchitz, ws)

        # check convergence
        if epoch % 10 == 0:
            dist = _compute_dist_subdiff(
                tensor_H_k, lmbd, residuals, S, ws)

            stop_crit_in = np.max(dist)
            if stop_crit_in <= tol_in:
                break
