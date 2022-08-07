import numpy as np
from numba import njit


@njit
def compute_p_obj(exp_yXw, w, alpha):
    datafit = np.sum(np.log(1 + exp_yXw))
    penalty = alpha * np.linalg.norm(w, ord=1)
    return datafit + penalty


@njit
def compute_d_obj(y, scaled_theta):
    """minus the Fenchel conjugate of datafit."""
    val = -scaled_theta / y
    return np.sum(-val * np.log(-val) - (1 - val) * np.log(1 - val))


@njit
def update_theta_exp_yXw(y, Xw, theta, exp_yXw):
    """Inplace update of theta and exp_yXw."""
    for i in range(len(y)):
        exp_yXw_i = np.exp(-y[i] * Xw[i])

        exp_yXw[i] = exp_yXw_i
        theta[i] = -y[i] * exp_yXw_i / (1 + exp_yXw_i)


def update_XTtheta(X, theta, XTtheta, ws):
    """Inplace update of XTtheta."""
    for j in ws:
        XTtheta[j] = X[:, j] @ theta


def update_phi_XTphi(scaled_theta, XTtheta, phi, XTphi, alpha, ws):
    """Inplace update of phi and XTphi."""
    # update as follows: max t for which
    #   new_phi = t scaled_theta + (1 - t) * phi is feasible
    #   <==> |constraint_j(new_phi)| = |X_j.T new_phi| <= alpha for j in ws
    best_t, t = 1.
    for j in ws:
        if XTtheta[j] <= alpha:
            pass

        if XTtheta[j] > 0:
            t = (alpha - XTphi[j]) / (XTtheta[j] - XTphi[j])
        else:
            t = (-alpha - XTphi[j]) / (XTtheta[j] - XTphi[j])

        best_t = min(t, best_t)

    for j in ws:
        XTphi[j] = best_t * XTtheta[j] + (1 - best_t) * XTphi[j]

    phi[:] = best_t * scaled_theta + (1 - best_t) * phi
