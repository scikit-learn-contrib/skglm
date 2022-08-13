import numpy as np
from numba import njit


LOGREG_LIPSCHITZ_CONST = 1.  # used in Blitz (weird)


@njit
def compute_primal_obj(exp_yXw, w, alpha):
    datafit = np.sum(np.log(1 + exp_yXw))
    penalty = alpha * np.linalg.norm(w, ord=1)
    return datafit + penalty


@njit
def compute_dual_obj(y, scaled_theta):
    """minus the Fenchel conjugate of datafit."""
    val = -scaled_theta / y
    return -np.sum(val * np.log(val) + (1 - val) * np.log(1 - val))


@njit
def update_theta_exp_yXw(y, Xw, theta, exp_yXw):
    """Inplace update of ``theta`` and ``exp_yXw``."""
    for i in range(len(y)):
        exp_yXw_i = np.exp(-y[i] * Xw[i])

        exp_yXw[i] = exp_yXw_i
        theta[i] = -y[i] * exp_yXw_i / (1 + exp_yXw_i)


@njit
def update_XTtheta(X, theta, XTtheta, ws):
    """Inplace update of ``XTtheta``."""
    for j in ws:
        XTtheta[j] = X[:, j] @ theta


@njit
def update_XTtheta_s(X_data, X_indptr, X_indices, theta, XTtheta, ws):
    """Inplace update of ``XTtheta``. Case ``X`` sparse."""
    for j in ws:
        tmp = 0.
        for i in range(X_indptr[j], X_indptr[j+1]):
            tmp += X_data[i] * theta[X_indices[i]]
        XTtheta[j] = tmp


@njit
def update_phi_XTphi(scaled_theta, scaled_XTtheta, phi, XTphi, alpha, ws):
    """Inplace update of ``phi`` and ``XTphi``."""
    # update as follows: max t for which
    #   new_phi = t * scaled_theta + (1 - t) * phi is feasible
    #   <==> |constraint_j(new_phi)| = |X_j.T new_phi| <= alpha for j in ws
    t = best_t = 1.
    for j in ws:
        if np.abs(scaled_XTtheta[j]) <= alpha:
            continue

        if scaled_XTtheta[j] >= 0:
            t = (alpha - XTphi[j]) / (scaled_XTtheta[j] - XTphi[j])
        else:
            t = (-alpha - XTphi[j]) / (scaled_XTtheta[j] - XTphi[j])

        best_t = min(t, best_t)

    # update XTphi and phi
    for j in ws:
        XTphi[j] = best_t * scaled_XTtheta[j] + (1 - best_t) * XTphi[j]

    phi[:] = best_t * scaled_theta + (1 - best_t) * phi


@njit
def compute_remaining_features(remaining_features, XTphi, w, norm2_X_cols, alpha, threshold):
    """Discard features whose scores are above ``threshold``."""
    features_scores = np.zeros(len(remaining_features))

    # score features
    for idx, j in enumerate(remaining_features):
        if w[j] != 0:
            score = 0.
        elif norm2_X_cols[j] == 0:
            score = np.inf
        else:
            score = (alpha - np.abs(XTphi[j])) / norm2_X_cols[j]

        features_scores[idx] = score

    # sort (could be improved)
    sorted_idx = np.argsort(features_scores)
    features_scores[:] = features_scores[sorted_idx]
    remaining_features[:] = remaining_features[sorted_idx]

    # discard features
    return remaining_features[features_scores <= threshold]
