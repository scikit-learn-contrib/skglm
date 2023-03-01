import numpy as np
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
