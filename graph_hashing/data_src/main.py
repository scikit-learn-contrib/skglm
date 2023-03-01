import utils
import networkx as nx
from hash_table import HashTable
import numpy as np


def generate_data(n_hashtable=10, n_nodes=10_000, n_supernodes=1_000, n_events=10_000):
    """Generate the graph hashing data.

    Parameters
    ----------
    n_hashtable: int, default=10
        Number of ``H_k`` matrices.

    n_nodes: int, default=10_000
        Specifies the number of columns in ``H_k``.
        It stands for the number of nodes in the original graph.

    n_supernodes: int, default=1_000
        Specifies the number of rows in ``H_k``.
        It stands for the number of nodes in when projecting
        the the original graph.

    n_event: int, default=10_000
        Controls the sparsity. The more it is high the less the problem is sparse.

    Returns
    -------
    tensor_H_k: array of shape (n_hashtable, n_nodes, n_supernodes)
        The List of H_k matrices as numpy array.

    tensor_S_k: array of shape (n_hashtable, n_supernodes, n_supernodes)
        The list of S_k matrices as numpy array.

    S_ground_truth: array of shape (n_nodes, n_nodes)
        The true matrix of interaction.
    """

    # number of H_k matrices
    number_of_hashtables = n_hashtable

    # size of H_k (n_nodes, n_supernodes)
    number_of_nodes = n_nodes
    number_supernodes = n_supernodes

    # number of interactions; the more interactions, the less sparse the matrix will be
    n_events = n_events

    tensor_H_k = np.zeros((n_hashtable, n_nodes, n_supernodes))
    tensor_S_k = np.zeros((n_hashtable, n_supernodes, n_supernodes))

    #######################
    ### data generation ###
    #######################

    # more supernodes = more precision and more time
    hashtables = []
    for i in range(number_of_hashtables):
        hashtables.append(utils.link_table(hashtable=HashTable(order=10, seed=42+i, multiseed=False),
                                           input_dimension=number_of_nodes, output_dimension=number_supernodes))

    mean_iet = 1  # mean inter event time
    average_k = 2  # average degree of the static graph, related to connectedness
    underlying_network = nx.gnp_random_graph(
        n=number_of_nodes, p=average_k / number_of_nodes, seed=42)
    # structure of the static graph
    events_list = utils.poisson_temporal_network(
        underlying_network, mean_iet=mean_iet, max_t=1000)
    events_list = events_list[:n_events]
    # list of interactions. Format [(node_i, node_j, timestep), ...]

    S_ground_truth = dense_matrix(events=events_list, hashtable=[
                                  i for i in range(number_of_nodes)], number_supernodes=number_of_nodes)

    for k in range(number_of_hashtables):
        H_kT = assignment_matrix(number_of_nodes=number_of_nodes,
                                 number_supernodes=number_supernodes, hashtable=hashtables[k])
        S_k = dense_matrix(events=events_list,
                           hashtable=hashtables[k], number_supernodes=number_supernodes)

        tensor_H_k[k] = H_kT.T
        tensor_S_k[k] = S_k

    # cast values as int
    tensor_H_k = tensor_H_k.astype(int)
    tensor_S_k = tensor_S_k.astype(int)
    S_ground_truth = S_ground_truth.astype(int)

    return tensor_H_k, tensor_S_k, S_ground_truth


def dense_matrix(events, hashtable, number_supernodes):
    """ Compute matrix S_k for k-th hashing in dense format (numpy array)"""
    S_k = utils.SI_process_parallel(
        events, link_table=hashtable, output_dimension=number_supernodes)
    return S_k


def sparse_matrix(events, hashtable, number_supernodes):  # sparse lil format
    """ Compute matrix S_k for k-th hashing in sparse lil format"""
    S_k = utils.SI_process_parallel_sparse(
        events, link_table=hashtable, output_dimension=number_supernodes)
    return S_k


def assignment_matrix(number_of_nodes, number_supernodes, hashtable):
    """ Compute transpose of H_k"""
    HkT = np.zeros((number_supernodes, number_of_nodes))
    for n_node in range(number_of_nodes):
        HkT[hashtable[n_node]][n_node] = 1
    return HkT


if __name__ == '__main__':
    pass
