import numpy as np

from graph_hashing.solvers import PGD, CD, FISTA, CD_WS
from graph_hashing.solvers.utils import compute_obj, compute_lmbd_max
from graph_hashing.data_src import generate_data


def test_data():
    n_hashtable = 10
    n_nodes = 10
    n_supernodes = 5
    n_events = 100

    tensor_H_k, tensor_S_k, S_true = generate_data(
        n_hashtable, n_nodes, n_supernodes, n_events)

    # check shape
    np.testing.assert_equal(
        tensor_H_k.shape, (n_hashtable, n_nodes, n_supernodes)
    )
    np.testing.assert_equal(
        tensor_S_k.shape, (n_hashtable, n_supernodes, n_supernodes)
    )
    np.testing.assert_equal(
        S_true.shape, (n_nodes, n_nodes)
    )

    # check type
    for arr in (tensor_H_k, tensor_H_k, S_true):
        np.testing.assert_equal(arr.dtype, int)


def test_solvers():
    n_hashtable = 3
    n_nodes = 10
    n_supernodes = 5

    reg = 1e-1

    # generate dummy data
    tensor_H_k, tensor_S_k, S_true = generate_data(
        n_hashtable, n_nodes, n_supernodes, n_events=100)

    lmbd = reg * compute_lmbd_max(tensor_H_k, tensor_S_k)

    # solve problem
    S_pgd, stop_crit_pgd = PGD(
        max_iter=10_000, tol=1e-12
    ).solve(tensor_H_k, tensor_S_k, lmbd)

    S_cd, stop_crit_cd = CD(
        max_iter=1000, tol=1e-12
    ).solve(tensor_H_k, tensor_S_k, lmbd)

    S_ws, stop_crit_ws = CD_WS(
        tol=1e-12
    ).solve(tensor_H_k, tensor_S_k, lmbd)

    S_fista, stop_crit_fista = FISTA(
        max_iter=10_000, tol=1e-12
    ).solve(tensor_H_k, tensor_S_k, lmbd)

    # check solver converges
    np.testing.assert_allclose(stop_crit_pgd, 0., atol=1e-12)
    np.testing.assert_allclose(stop_crit_fista, 0., atol=1e-12)
    np.testing.assert_allclose(stop_crit_cd, 0., atol=1e-12)
    np.testing.assert_allclose(stop_crit_ws, 0., atol=1e-12)

    # check solutions
    # despite converging solvers don't have the same solution
    np.testing.assert_allclose(
        compute_obj(S_pgd, tensor_H_k, tensor_S_k, lmbd),
        compute_obj(S_cd, tensor_H_k, tensor_S_k, lmbd),
    )
    np.testing.assert_allclose(
        compute_obj(S_pgd, tensor_H_k, tensor_S_k, lmbd),
        compute_obj(S_fista, tensor_H_k, tensor_S_k, lmbd),
    )
    np.testing.assert_allclose(
        compute_obj(S_pgd, tensor_H_k, tensor_S_k, lmbd),
        compute_obj(S_ws, tensor_H_k, tensor_S_k, lmbd),
    )


if __name__ == "__main__":
    pass
