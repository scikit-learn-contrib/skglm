from functools import lru_cache

import numpy as np
import numba
from numba import njit
from numba.experimental import jitclass
from numba import float32, float64

from numpy.linalg import norm
from sklearn.utils import check_random_state


def spec_to_float32(spec):
    """Convert a numba specification to an equivalent float32 one.

    Parameters
    ----------
    spec : list
        A list of (name, dtype) for every attribute of a jitclass.

    Returns
    -------
    spec32 : list
        A list of (name, dtype) for every attribute of a jitclass, where float64
        have been replaced by float32.
    """
    spec32 = []
    for name, dtype in spec:
        if dtype == float64:
            dtype32 = float32
        elif isinstance(dtype, numba.core.types.npytypes.Array):
            if dtype.dtype == float64:
                dtype32 = dtype.copy(dtype=float32)
            else:
                dtype32 = dtype
        else:
            raise ValueError(f"Unknown spec type {dtype}")
        spec32.append((name, dtype32))
    return spec32


@lru_cache()
def jit_cached_compile(klass, spec, to_float32=False):
    """Jit compile class and cache compilation.

    Parameters
    ----------
    klass : class
        Un instantiated Datafit or Penalty.

    spec : tuple
        A tuple of (name, dtype) for every attribute of a jitclass.

    to_float32 : bool, optional
        If ``True``converts float64 types to float32, by default False.

    Returns
    -------
    Instance of Datafit or penalty
        Return a jitclass.
    """
    if to_float32:
        spec = spec_to_float32(spec)

    return jitclass(spec)(klass)


def compiled_clone(instance, to_float32=False):
    """Compile instance to a jitclass.

    Parameters
    ----------
    instance : Instance of Datafit or Penalty
        Datafit or Penalty object.

    to_float32 : bool, optional
        If ``True``converts float64 types to float32, by default False.

    Returns
    -------
    Instance of Datafit or penalty
        Return a jitclass.
    """
    return jit_cached_compile(
        instance.__class__,
        instance.get_spec(),
        to_float32,
    )(**instance.params_to_dict())


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
def prox_MCP(value, stepsize, alpha, gamma):
    """Compute the proximal operator of stepsize * MCP penalty."""
    tau = alpha * stepsize
    g = gamma / stepsize  # what does g stand for ?
    if np.abs(value) <= tau:
        return 0.
    if np.abs(value) > g * tau:
        return value
    return np.sign(value) * (np.abs(value) - tau) / (1. - 1./g)


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


def make_correlated_data(
        n_samples=100, n_features=50, n_tasks=1, rho=0.6, snr=3,
        w_true=None, density=0.2, X_density=1, random_state=None):
    r"""Generate a linear regression with correlated design.

    The data are generated according to:

    .. math ::
        y = X w^* + \epsilon

    such that the signal to noise ratio is
    :math:`snr = \frac{||X w^*||}{||\epsilon||}`.

    The generated features have mean 0, variance 1 and the expected correlation
    structure

    .. math ::
        \mathbb E[x_i] = 0~, \quad \mathbb E[x_i^2] = 1  \quad
        and \quad \mathbb E[x_ix_j] = \rho^{|i-j|}

    Parameters
    ----------
    n_samples : int
        Number of samples in the design matrix.
    n_features : int
        Number of features in the design matrix.
    n_tasks : int
        Number of tasks.
    rho : float
        Correlation :math:`\rho` between successive features. The cross
        correlation :math:`C_{i, j}` between feature i and feature j will be
        :math:`\rho^{|i-j|}`. This parameter should be selected in
        :math:`[0, 1[`.
    snr : float or np.inf
        Signal-to-noise ratio.
    w_true : np.array, shape (n_features,) or (n_features, n_tasks)| None
        True regression coefficients. If None, a sparse array with standard
        Gaussian non zero entries is simulated.
    density : float
        Proportion of non zero elements in w_true if the latter is simulated.
    X_density : float in ]0, 1]
        Proportion of elements of X which are non-zero.
    random_state : int | RandomState instance | None (default)
        Determines random number generation for data generation. Use an int to
        make the randomness deterministic.

    Returns
    -------
    X : ndarray or CSC matrix, shape (n_samples, n_features)
        A design matrix with Toeplitz covariance.
    y : ndarray, shape (n_samples,) or (n_samples, n_tasks)
        Observation vector/matrix.
    w_true : ndarray, shape (n_features,) or (n_features, n_tasks)
        True regression vector/matrix of the model.
    """
    if not 0 <= rho < 1:
        raise ValueError("The correlation `rho` should be chosen in [0, 1[.")
    if not 0 < density <= 1:
        raise ValueError("The density should be chosen in ]0, 1].")
    if not 0 < X_density <= 1:
        raise ValueError("The density of X should be chosen in ]0, 1].")
    if snr < 0:
        raise ValueError("The snr should be chosen in [0, inf].")
    rng = check_random_state(random_state)
    nnz = int(density * n_features)

    if rho != 0:
        # X is generated cleverly using an AR model with reason corr and i
        # innovation sigma^2 = 1 - \rho ** 2: X[:, j+1] = rho X[:, j] + eps_j
        # where eps_j = sigma * rng.randn(n_samples)
        sigma = np.sqrt(1 - rho * rho)
        U = rng.randn(n_samples)

        X = np.empty([n_samples, n_features], order='F')
        X[:, 0] = U
        for j in range(1, n_features):
            U *= rho
            U += sigma * rng.randn(n_samples)
            X[:, j] = U
    else:
        X = rng.randn(n_samples, n_features)

    if X_density != 1:
        zeros = rng.binomial(n=1, size=X.shape, p=1 - X_density).astype(bool)
        X[zeros] = 0.
        from scipy import sparse
        X = sparse.csc_matrix(X)

    if w_true is None:
        w_true = np.zeros((n_features, n_tasks))
        support = rng.choice(n_features, nnz, replace=False)
        w_true[support, :] = rng.randn(nnz, n_tasks)
    else:
        if w_true.ndim == 1:
            w_true = w_true[:, None]

    Y = X @ w_true
    noise = rng.randn(n_samples, n_tasks)
    if snr not in [0, np.inf]:
        Y += noise / norm(noise) * norm(Y) / snr
    elif snr == 0:
        Y = noise

    if n_tasks == 1:
        return X, Y.flatten(), w_true.flatten()
    else:
        return X, Y, w_true


def grp_converter(groups, n_features):
    """Create group partition and group indices.

    Parameters
    ----------
    groups : int | list of ints | list of lists of ints
        Partition of features used in the penalty on `w`.
        If an int is passed, groups are contiguous blocks of features, of size
        `groups`.
        If a list of ints is passed, groups are assumed to be contiguous,
        group number `g` being of size `groups[g]`.
        If a list of lists of ints is passed, `groups[g]` contains the
        feature indices of the group number `g`.

    n_features : int
        Number of features.

    Returns
    -------
    grp_indices : array, shape (n_features,)
        The group indices stacked contiguously
        (e.g. [grp1_indices, grp2_indices, ...]).

    grp_ptr : array, shape (n_groups + 1,)
        The group pointers such that two consecutive elements delimit
        the indices of a group in ``grp_indices``.
    """
    if isinstance(groups, int):
        grp_size = groups
        if n_features % grp_size != 0:
            raise ValueError("n_features (%d) is not a multiple of the desired"
                             " group size (%d)" % (n_features, grp_size))
        n_groups = n_features // grp_size
        grp_ptr = grp_size * np.arange(n_groups + 1)
        grp_indices = np.arange(n_features)
    elif isinstance(groups, list) and isinstance(groups[0], int):
        grp_indices = np.arange(n_features)
        grp_ptr = np.cumsum(np.hstack([[0], groups]))
    elif isinstance(groups, list) and isinstance(groups[0], list):
        grp_sizes = np.array([len(ls) for ls in groups])
        grp_ptr = np.cumsum(np.hstack([[0], grp_sizes]))
        grp_indices = np.array([idx for grp in groups for idx in grp])
    else:
        raise ValueError("Unsupported group format.")
    return grp_indices.astype(np.int32), grp_ptr.astype(np.int32)


def check_group_compatible(obj):
    """Check whether ``obj`` is compatible with ``bcd_solver``.

    Parameters
    ----------
    obj : instance of BaseDatafit or BasePenalty
        Object to check.
    """
    obj_name = obj.__class__.__name__
    group_attrs = ('grp_ptr', 'grp_indices')

    for attr in group_attrs:
        if not hasattr(obj, attr):
            raise Exception(
                f"datafit and penalty must be compatible with 'bcd_solver'.\n"
                f"'{obj_name}' is not block-separable. "
                f"Missing '{attr}' attribute."
            )


def _alpha_max_group_lasso(X, y, grp_indices, grp_ptr, weights):
    n_samples = len(y)
    n_groups = len(grp_ptr) - 1
    alpha_max = 0.
    for g in range(n_groups):
        grp_g_indices = grp_indices[grp_ptr[g]: grp_ptr[g+1]]
        alpha_max = max(
            alpha_max,
            norm(X[:, grp_g_indices].T @ y) / (n_samples * weights[g])
        )
    return alpha_max


class AndersonAcceleration:
    """Abstraction of Anderson Acceleration.

    Extrapolate the asymptotic VAR ``w`` and ``Xw``
    based on ``K`` previous iterations.

    Parameters
    ----------
    K : int
        Number of previous iterates to consider for extrapolation.
    """

    def __init__(self, K):
        self.K, self.current_iter = K, 0
        self.arr_w_, self.arr_Xw_ = None, None

    def extrapolate(self, w, Xw):
        """Return w, Xw, and a bool indicating whether they were extrapolated."""
        if self.arr_w_ is None or self.arr_Xw_ is None:
            self.arr_w_ = np.zeros((w.shape[0], self.K+1))
            self.arr_Xw_ = np.zeros((Xw.shape[0], self.K+1))

        if self.current_iter <= self.K:
            self.arr_w_[:, self.current_iter] = w
            self.arr_Xw_[:, self.current_iter] = Xw
            self.current_iter += 1
            return w, Xw, False

        U = np.diff(self.arr_w_, axis=1)  # compute residuals

        # compute extrapolation coefs
        try:
            inv_UTU_ones = np.linalg.solve(U.T @ U, np.ones(self.K))
        except np.linalg.LinAlgError:
            return w, Xw, False
        finally:
            self.current_iter = 0

        # extrapolate
        C = inv_UTU_ones / np.sum(inv_UTU_ones)
        # floating point errors may cause w and Xw to disagree
        return self.arr_w_[:, 1:] @ C, self.arr_Xw_[:, 1:] @ C, True


@njit
def _prox_vec(w, z, penalty, lipschitz):
    # evaluate the vectorized proximal operator for the FISTA solver
    # lipschitz stands for global lipschitz constant
    n_features = w.shape[0]
    for j in range(n_features):
        w[j] = penalty.prox_1d(z[j], 1 / lipschitz, j)
    return w


@njit
def spectral_norm2(X_data, X_indptr, X_indices, n_samples,
                   max_iter=20, tol=1e-6):
    """Compute the squared spectral norm of sparse matrix ``X``.

    Find the largest eigenvalue of ``X @ X.T`` using the power method.

    Parameters
    ----------
    X_data : array, shape (n_elements,)
         `data` attribute of the sparse CSC matrix ``X``.

    X_indptr : array, shape (n_features + 1,)
         `indptr` attribute of the sparse CSC matrix ``X``.

    X_indices : array, shape (n_elements,)
         `indices` attribute of the sparse CSC matrix ``X``.

    n_samples : int
        number of rows of ``X``.

    Returns
    -------
    eigenvalue : float
        The largest eigenvalue of ``X.T @ X``, aka the squared spectral norm of ``X``.

    References
    ----------
    .. [1] Alfio Quarteroni, Riccardo Sacco, Fausto Saleri "Numerical Mathematics",
        chapiter 5, page 192-195.
    """
    # tol is squared as we evaluate the square of inequality (5.25) in ref [1]
    tol = tol ** 2

    # init vec with norm(vec) == 1.
    eigenvector = np.random.randn(n_samples)
    eigenvector /= norm(eigenvector)
    eigenvalue = 1.

    for _ in range(max_iter):
        vec = _XXT_dot_vec(X_data, X_indptr, X_indices, eigenvector, n_samples)
        norm_vec = norm(vec)
        eigenvalue = vec @ eigenvector

        # norm(X @ X.T @ eigenvector - eigenvalue * eigenvector) <= tol
        # inequality is squared
        if norm_vec ** 2 - eigenvalue ** 2 <= tol:
            break

        eigenvector = vec / norm_vec

    return eigenvalue


@njit
def _XXT_dot_vec(X_data, X_indptr, X_indices, vec, n_samples):
    # computes X @ X.T @ vec, with X csc encoded
    return _X_dot_vec(X_data, X_indptr, X_indices,
                      _XT_dot_vec(X_data, X_indptr, X_indices, vec),
                      n_samples)


@njit
def _X_dot_vec(X_data, X_indptr, X_indices, vec, n_samples):
    # compute X @ vec, with X csc encoded
    result = np.zeros(n_samples)

    # loop over features
    for j in range(len(X_indptr) - 1):
        if vec[j] == 0:
            continue

        col_j_rows_idx = slice(X_indptr[j], X_indptr[j+1])
        result[X_indices[col_j_rows_idx]] += vec[j] * X_data[col_j_rows_idx]

    return result


@njit
def _XT_dot_vec(X_data, X_indptr, X_indices, vec):
    # compute X.T @ vec, with X csc encoded
    n_features = len(X_indptr) - 1
    result = np.zeros(n_features)

    for j in range(n_features):
        for idx in range(X_indptr[j], X_indptr[j+1]):
            result[j] += X_data[idx] * vec[X_indices[idx]]

    return result


if __name__ == '__main__':
    from scipy.sparse import csc_matrix, random
    import time
    n_samples, n_features = 500, 600
    A = random(n_samples, n_features, density=0.5, format='csc')

    X = csc_matrix(A)
    X_dense = X.toarray()

    # cache numba compilation
    M = random(5, 7, density=0.9, format='csc')
    spectral_norm2(M.data, M.indptr, M.indices, 5)

    start = time.time()
    spectral_norm2(X.data, X.indptr, X.indices, n_samples)
    end = time.time()
    print("our: ", end - start)

    start = time.time()
    norm(X_dense, ord=2) ** 2
    end = time.time()
    print("np: ", end - start)
