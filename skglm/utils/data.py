import numpy as np
import scipy.sparse
from numpy.linalg import norm
from sklearn.utils import check_random_state
from sklearn.preprocessing import StandardScaler


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


def make_dummy_survival_data(n_samples, n_features, normalize=False,
                             X_density=1., random_state=None):
    """Generate a random dataset for survival analysis.

    The design matrix ``X`` is generated according to standard normal, the vector of
    time ``tm`` is chosen according to a Weibull(1, 1) (aka. Exponential), and the
    vector of censorship ``s`` is drawn from a Bernoulli with parameter ``0.5``.

    Parameters
    ----------
    n_samples : int
        Number of samples in the design matrix.

    n_features : int
        Number of features in the design matrix.

    normalize : bool, default=False
        If ``True``, features are centered and divided by their
        standard deviation. This argument is ineffective when ``X_density < 1``.

    X_density : float, default=1
        The density, proportion of non zero elements, of the design matrix ``X``.
        X_density must be in ``(0, 1]``.

    random_state : int, default=None
        Determines random number generation for data generation.

    Returns
    -------
    tm : array-like, shape (n_samples,)
        The vector of recording the time of event occurrences

    s : array-like, shape (n_samples,)
        The vector of indicating samples censorship

    X : array-like, shape (n_samples, n_features)
        The matrix of predictors. If ``density < 1``, a CSC sparse matrix is returned.
    """
    rng = np.random.RandomState(random_state)

    if X_density == 1.:
        X = rng.randn(n_samples, n_features).astype(float, order='F')
    else:
        X = scipy.sparse.rand(
            n_samples, n_features, density=X_density, format="csc", dtype=float)

    tm = rng.weibull(a=1, size=n_samples)
    s = rng.choice(2, size=n_samples).astype(float)

    if normalize and X_density == 1.:
        X = StandardScaler().fit_transform(X)

    return tm, s, X


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
