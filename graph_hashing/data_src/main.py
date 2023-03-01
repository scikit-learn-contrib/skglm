import utils
import networkx as nx
from hash_table import HashTable
import numpy as np


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

    number_of_hashtables = 10  # number of small matrices
    n_events = 10000  # number of interactions; the more interactions, the less sparse the matrix will be
    number_of_nodes = 10000
    number_supernodes = 1000
    #  more supernodes = more precision and more time
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
