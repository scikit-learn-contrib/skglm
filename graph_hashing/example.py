import numpy

from graph_hashing.solvers import PGD, CD
from graph_hashing.solvers.fista import FISTA
from graph_hashing.solvers.utils import compute_lmbd_max
from graph_hashing.data_src import generate_data


reg = 1e-1
n_H_k, n_nodes, n_supernodes = 10, 100, 20

# tensore_H_k has shape (n_H_k, n_nodes, n_supernodes)
# tensore_S_k has shape (n_H_k, n_supernodes, n_supernodes)
tensor_H_k, tensor_S_k, _ = generate_data(n_H_k, n_nodes, n_supernodes, n_events=100)


########################
###  compute lmd max ###
########################
lmbd_max = compute_lmbd_max(tensor_H_k, tensor_S_k)


########################
###  solve in prob   ###
########################
lmbd = reg * lmbd_max

S, stop_crit = CD(verbose=1, max_iter=100, tol=1e-15).solve(
    tensor_H_k, tensor_S_k, lmbd)

# print support of solution
print((S != 0).sum())
