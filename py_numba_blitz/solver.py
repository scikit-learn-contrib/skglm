import numpy as np

from py_numba_blitz.utils import(compute_primal_obj, compute_dual_obj,
                                 compute_remaining_features,
                                 update_XTtheta, update_phi_XTphi,
                                 update_theta_exp_yXw, LOGREG_LIPSCHITZ_CONST)


def py_blitz(alpha, X, y, p0=100, max_iter=20, max_epochs=100):
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

    remaining_features = np.arange(n_features)
    norm2_X_cols = np.zeros(n_features)

    # init vars
    update_theta_exp_yXw(y, Xw, theta, exp_yXw)
    for j in range(n_features):
        norm2_X_cols[j] = np.linalg.norm(X[:, j], ord=2)

    for t in range(max_iter):
        update_XTtheta(X, theta, XTtheta, remaining_features)
        update_phi_XTphi(theta, XTtheta, phi, XTphi, alpha, remaining_features)

        p_obj = compute_primal_obj(exp_yXw, w, alpha)
        d_obj = compute_dual_obj(y, phi)
        gap = p_obj - d_obj

        threshold = np.sqrt(2 * gap / LOGREG_LIPSCHITZ_CONST)
        remaining_features = compute_remaining_features(remaining_features, XTphi,
                                                        w, norm2_X_cols, alpha, threshold)

        ws_size = min(
            max(2*np.sum(w != 0), p0),
            len(remaining_features)
        )

        print(
            f'primal obj: {p_obj}\n'
            f'dual obj: {d_obj}\n'
            f'gap: {gap}\n'
            f'ws size: {ws_size}\n'
            f'ws: {remaining_features[:ws_size]}'
        )

        for epoch in range(max_epochs):
            pass
    return
