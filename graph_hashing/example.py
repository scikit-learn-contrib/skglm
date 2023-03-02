import numpy as np
from numpy.linalg import norm

from graph_hashing.data_src import generate_data
from graph_hashing.solvers import PGD


reg = 1e-1
n_H_k, n_nodes, n_supernodes = 10, 100, 10

# tensore_H_k has shape (n_H_k, n_nodes, n_supernodes)
# tensore_S_k has shape (n_H_k, n_supernodes, n_supernodes)
tensor_H_k, tensor_S_k, _ = generate_data(n_H_k, n_nodes, n_supernodes, n_events=100)


########################
###  compute lmd max ###
########################
grad_zero = np.zeros((n_nodes, n_nodes))

for H_k, S_k in zip(tensor_H_k, tensor_S_k):
    grad_zero -= H_k @ S_k @ H_k.T

lmbd_max = norm(grad_zero.flatten(), ord=np.inf)


########################
###  solve in prob   ###
########################
lmbd = reg * lmbd_max

S, stop_crit = PGD(verbose=1, max_iter=500).solve(tensor_H_k, tensor_S_k, lmbd)

print((S != 0).sum())