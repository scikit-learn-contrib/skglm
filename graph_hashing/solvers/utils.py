import numpy as np
from numpy.linalg import norm


def compute_obj(S, tensor_H_k, tensor_S_k, lmbd):

    datafit_val = 0.
    # compute datafit
    for k in range(len(tensor_H_k)):
        residual_k = compute_residual_k(S, tensor_H_k, tensor_S_k, k)
        datafit_val += 0.5 * norm(residual_k) ** 2

    # compute penalty
    penalty_val = lmbd * np.abs(S).sum()

    return datafit_val + penalty_val


def compute_residual_k(S, tensor_H_k, tensor_S_k, k):
    S_k = tensor_S_k[k]
    H_k = tensor_H_k[k]

    return S_k - H_k.T @ S @ H_k


def validate_input(tensor_H_k, tensor_S_k):
    # TODO: validate type shape of parameters

    # cast as float
    tensor_H_k = tensor_H_k.astype(float)
    tensor_S_k = tensor_S_k.astype(float)

    return tensor_H_k, tensor_S_k
