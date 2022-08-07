import numpy as np

from py_numba_blitz.utils import(compute_primal_obj, compute_dual_obj,
                                 update_XTtheta, update_phi_XTphi,
                                 update_theta_exp_yXw)


def py_blitz(alpha, X, y, max_iter, max_epochs):
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
    update_theta_exp_yXw(y, Xw, theta, exp_yXw)

    for t in range(max_iter):
        p_obj = compute_primal_obj(exp_yXw, w, alpha)

        ws = np.arange(n_features)
        update_XTtheta(X, theta, XTtheta, ws)
        update_phi_XTphi(theta, XTtheta, phi, XTphi, alpha, ws)

        d_obj = compute_dual_obj(y, phi)
        gap = p_obj - d_obj

        print(
            f'primal obj: {p_obj}\n'
            f'dual obj: {d_obj}\n'
            f'gap: {gap}'
        )
        for epoch in range(max_epochs):
            pass
    return
