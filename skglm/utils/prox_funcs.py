import numpy as np
from numba import njit
from numpy.linalg import norm


@njit
def ST(x, u, positive=False):
    """Soft-thresholding of scalar x at level u."""
    if x > u:
        return x - u
    elif x < - u and not positive:
        return x + u
    else:
        return 0.


@njit
def ST_vec(x, u):
    """Entrywise soft-thresholding of array x at level u."""
    return np.sign(x) * np.maximum(0., np.abs(x) - u)


@njit
def proj_L2ball(u):
    """Project input on L2 unit ball."""
    norm_u = norm(u)
    if norm_u <= 1:
        return u
    return u / norm_u


@njit
def BST(x, u):
    """Block soft-thresholding of vector x at level u."""
    norm_x = norm(x)
    if norm_x < u:
        return np.zeros_like(x)
    else:
        return (1 - u / norm_x) * x


@njit
def box_proj(x, low, up):
    """Projection of scalar x onto [low, up] interval."""
    if x > up:
        return up
    elif x < low:
        return low
    else:
        return x


@njit
def value_MCP(w, alpha, gamma):
    """Compute the value of MCP."""
    s0 = np.abs(w) < gamma * alpha
    value = np.full_like(w, gamma * alpha ** 2 / 2.)
    value[s0] = alpha * np.abs(w[s0]) - w[s0]**2 / (2 * gamma)
    return np.sum(value)


@njit
def value_weighted_MCP(w, alpha, gamma, weights):
    """Compute the value of the weighted MCP."""
    s0 = np.abs(w) < gamma * alpha
    value = np.full_like(w, gamma * alpha ** 2 / 2.)
    value[s0] = alpha * np.abs(w[s0]) - w[s0]**2 / (2 * gamma)
    return np.sum(weights * value)


@njit
def prox_MCP(value, stepsize, alpha, gamma, positive=False, weight=1.):
    """Compute the proximal operator of stepsize * weight MCP penalty."""
    wstepsize = weight * stepsize  # weighted stepsize
    if (np.abs(value) <= alpha * wstepsize) or (positive and value <= 0.):
        return 0.
    if np.abs(value) > alpha * gamma:
        return value
    return np.sign(value) * (np.abs(value) - alpha * wstepsize) / (1. - wstepsize/gamma)


@njit
def value_SCAD(w, alpha, gamma):
    """Compute the value of the SCAD penalty at w."""
    value = np.full_like(w, alpha ** 2 * (gamma + 1) / 2)
    for j in range(len(w)):
        if np.abs(w[j]) <= alpha:
            value[j] = alpha * np.abs(w[j])
        elif np.abs(w[j]) <= alpha * gamma:
            value[j] = (
                2 * gamma * alpha * np.abs(w[j])
                - w[j] ** 2 - alpha ** 2) / (2 * (gamma - 1))
    return np.sum(value)


@njit
def prox_SCAD(value, stepsize, alpha, gamma):
    """Compute the proximal operator of stepsize * SCAD penalty."""
    # A general iterative shrinkage and thresholding algorithm for non-convex
    # regularized optimization problems, (Gong et al., 2013, Appendix)
    # see: http://proceedings.mlr.press/v28/gong13a.pdf
    tau = gamma * alpha
    x_1 = max(0, np.abs(value) - alpha * stepsize)
    x_2 = ((gamma - 1) * np.abs(value) - stepsize * tau) / (
        gamma - 1 - stepsize)
    x_2 = abs(x_2)
    x_3 = abs(value)
    x_s = [x_1, x_2, x_3]

    objs = np.array([(0.5 / stepsize) * (x - np.abs(value)) ** 2 + value_SCAD(
        np.array([x]), alpha, gamma) for x in x_s])
    return np.sign(value) * x_s[np.argmin(objs)]


@njit
def BST_vec(x, u, grp_size):
    """Vectorized block soft-thresholding of vector x at level u."""
    norm_grp = norm(x.reshape(-1, grp_size), axis=1)
    scaling = np.maximum(1 - u / norm_grp, 0)
    return (x.reshape(-1, grp_size) * scaling[:, None]).reshape(x.shape[0])


@njit
def prox_05(x, u):
    """Scalar version of the prox of L0.5 norm."""
    t = (3./2.) * u ** (2./3.)
    if np.abs(x) < t:
        return 0.
    return x * (2./3.) * (1 + np.cos((2./3.) * np.arccos(
        -(3.**(3./2.)/4.) * u * np.abs(x)**(-3./2.))))


@njit
def prox_block_2_05(x, u):
    """Proximal operator of block L0.5 penalty."""
    norm_x = norm(x, ord=2)
    return (prox_05(norm_x, u) / norm_x) * x


@njit
def prox_2_3(x, u):
    """Proximal operator of block L2/3 penalty."""
    t = 2.*(2./3. * u)**(3./4.)
    if np.abs(x) < t:
        return 0.
    z = (x**2 / 16 + np.sqrt(x**4/256 - 8 * u**3 / 729))**(1./3.) + (
        x**2 / 16 - np.sqrt(x**4/256 - 8 * u**3 / 729))**(1./3.)
    res = np.sign(x) * 1./8. * (
        np.sqrt(2.*z) + np.sqrt(2.*np.abs(x)/np.sqrt(2.*z)-2.*z))**3
    return res


@njit
def _prox_vec(w, z, penalty, step):
    # evaluate the full proximal operator
    n_features = w.shape[0]
    for j in range(n_features):
        w[j] = penalty.prox_1d(z[j], step, j)
    return w


@njit
def prox_SLOPE(z, alphas):
    """Fast computation for proximal operator of SLOPE.

    Extracted from:
    https://github.com/agisga/grpSLOPE/blob/master/src/proxSortedL1.c

    Parameters
    ----------
    z : array, shape (n_features,)
        Non-negative coefficient vector sorted in non-increasing order.

    alphas : array, shape (n_features,)
        Regularization hyperparameter sorted in non-increasing order.
    """
    n_features = z.shape[0]
    x = np.empty(n_features)

    k = 0
    idx_i = np.empty((n_features,), dtype=np.int64)
    idx_j = np.empty((n_features,), dtype=np.int64)
    s = np.empty((n_features,), dtype=np.float64)
    w = np.empty((n_features,), dtype=np.float64)

    for i in range(n_features):
        idx_i[k] = i
        idx_j[k] = i
        s[k] = z[i] - alphas[i]
        w[k] = s[k]

        while k > 0 and w[k - 1] <= w[k]:
            k -= 1
            idx_j[k] = i
            s[k] += s[k+1]
            w[k] = s[k] / (i - idx_i[k] + 1)

        k += 1

    for j in range(k):
        d = w[j]
        d = 0 if d < 0 else d
        for i in range(idx_i[j], idx_j[j] + 1):
            x[i] = d

    return x


@njit
def log_sum_vec(x, eps):
    return np.log(1 + np.exp(x) / eps)


@njit
def _root_prox_log_vec(x, alpha, eps):
    return (x - eps) / 2. + np.sqrt(((x + eps) ** 2) / 4 - alpha)


@njit
def _log_sum_prox_val(x, z, alpha, eps):
    return ((x - z) ** 2) / (2 * alpha) + log_sum_vec(x, eps)


@njit
def _r(x, alpha, eps):
    """r as defined in Prater-Bennette et al. (2021)."""
    r_z = _log_sum_prox_val(_root_prox_log_vec(x, alpha, eps), x, alpha, eps)
    r_0 = _log_sum_prox_val(0, x, alpha, eps)
    return r_z - r_0


@njit
def _find_root_by_bisection(a, b, alpha, eps, tol=1e-8):
    """Find root of function func in interval [a, b] by bisection."""
    while b - a > tol:
        c = (a + b) / 2.
        if _r(a, alpha, eps) * _r(c, alpha, eps) < 0:
            b = c
        else:
            a = c
    return c


@njit
def prox_log_sum(x, alpha, eps):
    """Proximal operator of log-sum penalty.

    Parameters
    ----------
    x : float
        Coefficient.

    alpha : float
        Regularization hyperparameter.

    eps : float
        Curvature hyperparameter.

    Reference
    ---------
        https://www.researchgate.net/profile/Erin-Tripp/publication/ \
        349804616_The_Proximity_Operator_of_the_Log-Sum_Penalty/links/ \
        60914684a6fdccaebd08e9ff/The-Proximity-Operator-of-the-Log-Sum-Penalty.pdf \
        ?origin=publication_detail
    """
    if np.sqrt(alpha) <= eps:
        if abs(x) <= alpha / eps:
            return 0.
        else:
            return np.sign(x) * _root_prox_log_vec(abs(x), alpha, eps)
    else:
        a = 2 * np.sqrt(alpha) - eps
        b = alpha / eps
        # f is continuous and f(a) * f(b) < 0, the root can be found by bisection
        x_star = _find_root_by_bisection(a, b, alpha, eps)
        if abs(x) <= x_star:
            return 0.
        else:
            return np.sign(x) * _root_prox_log_vec(abs(x), alpha, eps)
