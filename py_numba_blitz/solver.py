import numpy as np

from py_numba_blitz.utils import(compute_p_obj, compute_d_obj,
                                 update_theta_exp_yXw)


def log_reg(alpha, X, y, max_iter, max_epochs):
    r"""Solve Logistic Regression.

    Objective:
        \sum_{i=1}^n log(1 + e^{-y_i (Xw)_i}) + \alpha \sum_{j=1}^p |w_j| 
    """
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    Xw = np.zeros(n_samples)
    exp_yXw = np.zeros(n_samples)  # np.exp(-y * Xw)
    # dual vars
    theta = np.zeros(n_samples)
    phi = np.zeros(n_samples)
    XTtheta = np.zeros(n_features)
    XTphi = np.zeros(n_features)

    # init vars

    for t in range(max_iter):

        for epoch in range(max_epochs):
            pass
    return
