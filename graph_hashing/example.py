import numpy as np
from numpy.linalg import norm

from graph_hashing.data_src import generate_data
from graph_hashing.solvers import PGD


reg = 1e-1
n_H_k, n_nodes, n_supernodes = 10, 1000, 100
tensor_H_k, tensor_S_k, _ = generate_data(n_H_k, n_nodes, n_supernodes)

########################
###  compute lmd max ###
########################
residual = tensor_S_k.sum(axis=0)
grad_zero = np.zeros((n_nodes, n_nodes))

for H_k in tensor_H_k:
    grad_zero -= H_k @ residual @ H_k.T

lmbd_max = norm(grad_zero.flatten(), ord=np.inf)


########################
###  solve in prob   ###
########################
lmbd = reg * lmbd_max

PGD(verbose=1).solve(tensor_H_k, tensor_S_k, lmbd)
