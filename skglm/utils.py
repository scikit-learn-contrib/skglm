import numpy as np
from numba import njit

from numpy.linalg import norm
from sklearn.utils import check_random_state


@njit
def ST(x, u):
    """Soft-thresholding of scalar x at level u."""
    if x > u:
        return x - u
    elif x < - u:
        return x + u
    else:
        return 0.


@njit
def ST_vec(x, u):
    """Entrywise soft-thresholding of array x at level u."""
    return np.sign(x) * np.maximum(0., np.abs(x) - u)


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
