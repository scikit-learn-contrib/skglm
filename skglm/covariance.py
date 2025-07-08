# License: BSD 3 clause

import numpy as np
from numba import njit
from scipy.linalg import pinvh
from sklearn.base import BaseEstimator

from skglm.utils.prox_funcs import ST
from skglm.penalties.separable import L0_5


# GLasso uses this minimal Numba-compiled CD kernel because it is noticeably
# faster than the generic GramCD/epoch solvers (see discussion:
# https://github.com/scikit-learn-contrib/skglm/pull/280#discussion_r2172199185).

@njit
def barebones_cd_gram(H, q, x, alpha, weights, max_iter=100, tol=1e-4):
    """Solve min .5 * x.T H x + q.T @ x + alpha * norm(x, 1).

    with *H* symmetric positive definite.

    This variant is kept minimal (no penalty abstraction, no greedy feature
    selection) to minimise Python overhead and maximise Numba optimisation.  It
    consistently outperforms the generic `GramCD`/`epoch` variants in the
    GraphicalLasso context (see link above).
    """
    dim = H.shape[0]
    lc = np.zeros(dim)
    for j in range(dim):
        lc[j] = H[j, j]
    Hx = np.dot(H, x)

    for _ in range(max_iter):
        max_delta = 0  # max coeff change
        for j in range(dim):
            x_j_prev = x[j]
            x[j] = ST(x[j] - (Hx[j] + q[j]) / lc[j], alpha*weights[j] / lc[j])
            max_delta = max(max_delta, np.abs(x_j_prev - x[j]))
            if x_j_prev != x[j]:
                Hx += (x[j] - x_j_prev) * H[j]
        if max_delta <= tol:
            break

    return x


class GraphicalLasso(BaseEstimator):
    r"""Block Coordinate Descent (BCD) solver for the Graphical Lasso (GLasso) problem.

    Implements the sparse inverse covariance estimation algorithm from Friedman et al.
    (2008) and its weighted/primal variant from Mazumder et al. (2012), supporting both
    primal and dual coordinate descent.

    The problem is defined as:

    .. math::

        \Theta^* = \underset{\Theta \succ 0}{\arg\min}\ -\log\det(\Theta)
        + \langle S, \Theta \rangle + \alpha \sum_{i \ne j} w_{ij} |\Theta_{ij}|

    where :math:`S` is the empirical covariance matrix, :math:`\Theta` the
    precision matrix, :math:`\alpha` the regularisation strength, and
    :math:`w_{ij}` optional positive weights (uniform when ``weights=None``).

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength.
    weights : ndarray or None, default=None
        Symmetric matrix of penalty weights, or None for uniform weights.
    algo : {'dual', 'primal'}, default='dual'
        Algorithm variant to use.
    max_iter : int, default=100
        Maximum number of coordinate descent iterations.
    tol : float, default=1e-8
        Convergence tolerance.
    warm_start : bool, default=False
        Use previous solution as initialization (primal only).
    inner_tol : float, default=1e-4
        Tolerance for inner solver.
    verbose : bool, default=False
        Print convergence info.

    Attributes
    ----------
    precision_ : ndarray
        Estimated precision (inverse covariance) matrix.
    covariance_ : ndarray
        Estimated covariance matrix.
    n_iter_ : int
        Number of iterations run.

    References
    ----------
    .. [1] Friedman et al., Biostatistics, 2008.
           https://doi.org/10.1093/biostatistics/kxm045
    .. [2] Mazumder et al., Electron. J. Statist., 2012.
           https://doi.org/10.1214/12-EJS740
    """

    def __init__(self,
                 alpha=1.,
                 weights=None,
                 algo="dual",
                 max_iter=100,
                 tol=1e-8,
                 warm_start=False,
                 inner_tol=1e-4,
                 verbose=False,
                 solver="barebones"
                 ):
        self.alpha = alpha
        self.weights = weights
        self.algo = algo
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.inner_tol = inner_tol
        self.verbose = verbose
        self.solver = solver

    def fit(self, X, y=None, mode='empirical'):
        """Fit the GraphicalLasso model.

        Parameters
        ----------
        X : ndarray
            Data matrix (n_samples, n_features) if mode='empirical', or
            covariance matrix (n_features, n_features) if mode='precomputed'.
        mode : {'empirical', 'precomputed'}, default='empirical'
            If 'empirical', X is treated as a data matrix and the empirical
            covariance is computed.
            If 'precomputed', X is treated as a covariance matrix.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("Input must be a 2D array.")
        if mode == 'precomputed':
            S = X
        else:
            S = np.cov(X, bias=True, rowvar=False)
        p = S.shape[-1]
        indices = np.arange(p)

        if self.weights is None:
            Weights = np.ones((p, p))
        else:
            Weights = self.weights
            if not np.allclose(Weights, Weights.T):
                raise ValueError("Weights should be symmetric.")

        if self.warm_start and hasattr(self, "precision_"):
            if self.algo == "dual":
                raise ValueError(
                    "dual does not support warm start for now.")
            Theta = self.precision_
            W = self.covariance_
        else:
            # Ensure W is SPD
            eigvals = np.linalg.eigvalsh(S)
            min_eig = eigvals[0]
            eps = max(self.tol, np.finfo(float).eps)
            ridge = max(0.0, -min_eig + eps)
            W = S + ridge * np.eye(p)

            Theta = pinvh(W)

        W_11 = np.copy(W[1:, 1:], order="C")
        eps = np.finfo(np.float64).eps
        it = 0
        Theta_old = Theta.copy()

        for it in range(self.max_iter):
            Theta_old = Theta.copy()

            for col in range(p):
                if self.algo == "primal":
                    indices_minus_col = np.concatenate(
                        [indices[:col], indices[col + 1:]])
                    _11 = indices_minus_col[:, None], indices_minus_col[None]
                    _12 = indices_minus_col, col
                    _22 = col, col

                elif self.algo == "dual":
                    if col > 0:
                        di = col - 1
                        W_11[di] = W[di][indices != col]
                        W_11[:, di] = W[:, di][indices != col]
                    else:
                        W_11[:] = W[1:, 1:]

                s_12 = S[col, indices != col]

                if self.algo == "dual":
                    beta_init = (Theta[indices != col, col] /
                                 (Theta[col, col] + 1000 * eps))
                    Q = W_11

                elif self.algo == "primal":
                    inv_Theta_11 = (W[_11] -
                                    np.outer(W[_12],
                                             W[_12])/W[_22])
                    Q = inv_Theta_11
                    beta_init = Theta[indices != col, col] * S[col, col]
                else:
                    raise ValueError(f"Unsupported algo {self.algo}")

                # use the specialised bare-bones gram-cd solver
                beta = barebones_cd_gram(
                    Q,
                    s_12,
                    x=beta_init,
                    alpha=self.alpha,
                    weights=Weights[indices != col, col],
                    tol=self.inner_tol,
                    max_iter=self.max_iter,
                )

                if self.algo == "dual":
                    w_12 = -np.dot(W_11, beta)
                    W[col, indices != col] = w_12
                    W[indices != col, col] = w_12

                    Theta[col, col] = 1 / \
                        (W[col, col] + np.dot(beta, w_12))
                    Theta[indices != col, col] = beta*Theta[col, col]
                    Theta[col, indices != col] = beta*Theta[col, col]

                else:  # primal
                    s_22 = S[col, col]

                    # Updating Theta
                    theta_12 = beta / s_22
                    Theta[indices != col, col] = theta_12
                    Theta[col, indices != col] = theta_12
                    Theta[col, col] = (1/s_22 +
                                       theta_12 @
                                       inv_Theta_11 @
                                       theta_12)
                    theta_22 = Theta[col, col]

                    # Updating W
                    W[col, col] = (1/(theta_22 -
                                      theta_12 @
                                      inv_Theta_11 @
                                      theta_12))
                    w_22 = W[col, col]

                    w_12 = (-w_22 * inv_Theta_11 @ theta_12)
                    W[indices != col, col] = w_12
                    W[col, indices != col] = w_12

                    # Maybe W_11 can be done smarter ?
                    W[_11] = (inv_Theta_11 +
                              np.outer(w_12,
                                       w_12)/w_22)

            if np.linalg.norm(Theta - Theta_old) < self.tol:
                if self.verbose:
                    print(f"Weighted Glasso converged at CD epoch {it + 1}")
                break
        else:
            if self.verbose:
                print(
                    f"Not converged at epoch {it + 1}, "
                    f"diff={np.linalg.norm(Theta - Theta_old):.2e}"
                )
        self.precision_, self.covariance_ = Theta, W
        self.n_iter_ = it + 1

        return self


class AdaptiveGraphicalLasso(BaseEstimator):
    r"""Adaptive version of the Graphical Lasso with non-convex penalties.

    Solves non-convex penalty variations using an iterative reweighting strategy
    (Candès et al., 2007).

    At each reweighting iteration, it solves

    .. math::

        \Theta^* = \underset{\Theta \succ 0}{\arg\min}\ -\log\det(\Theta)
        + \langle S, \Theta \rangle + \alpha \sum_{i \ne j} w_{ij}^{(k)} |\Theta_{ij}|

    where the weights are updated as

    .. math::

        w_{ij}^{(k)} = \frac{1}{\epsilon} \quad \text{if } \Theta_{ij}^{(k-1)} = 0

    .. math::

        w_{ij}^{(k)} = |p'(\Theta_{ij}^{(k-1)})| \quad \text{otherwise}

    where :math:`p'(\cdot)` is the derivative of the non-convex penalty function.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization parameter controlling sparsity.
    eps : float, default=1e-10
        Small value for handling exactly zero elements in reweighting.
        Controls numerical stability of the adaptive algorithm.
    n_reweights : int, default=5
        Number of reweighting iterations.
    max_iter : int, default=1000
        Maximum iterations for inner solver.
    tol : float, default=1e-8
        Convergence tolerance.
    warm_start : bool, default=False
        Whether to use warm start.
    penalty : Penalty object, default=L0_5(1.)
        Non-convex penalty function. Must have a 'derivative' method.
        The penalty's alpha parameter should typically match this class's alpha.
    verbose : bool, default=False
        Whether to print verbose output.

    Attributes
    ----------
    precision_ : ndarray
        Estimated precision (inverse covariance) matrix.
    covariance_ : ndarray
        Estimated covariance matrix.
    n_iter_ : int
        Number of iterations run.

    References
    ----------
    .. [1] Candès et al., 2007.
    """

    def __init__(
        self,
        alpha=1.,
        eps=1e-10,  # Handles adaptive reweighting of zero elements
        n_reweights=5,
        max_iter=1000,
        tol=1e-8,
        warm_start=False,
        penalty=L0_5(1.),
        verbose=False,
    ):
        if not hasattr(penalty, 'derivative'):
            raise ValueError("penalty must have a 'derivative' method")

        self.alpha = alpha
        self.eps = eps
        self.n_reweights = n_reweights
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.penalty = penalty
        self.verbose = verbose

    def fit(self, X, y=None, mode='empirical'):
        """Fit the AdaptiveGraphicalLasso model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or (n_features, n_features)
            Data matrix or empirical covariance matrix. If a data matrix is provided,
            the empirical covariance is computed. If a square matrix is provided,
            it is always treated as a covariance matrix.
        mode : {'empirical', 'precomputed'}, default='empirical'
            If 'empirical', X is treated as a data matrix and the empirical
            covariance is computed.
            If 'precomputed', X is treated as a covariance matrix.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("Input must be a 2D array.")

        if mode == 'precomputed':
            S = X
        else:
            S = np.cov(X, bias=True, rowvar=False)
        glasso = GraphicalLasso(
            alpha=self.alpha,
            algo="primal",
            max_iter=self.max_iter,
            tol=self.tol,
            warm_start=self.warm_start,
            verbose=self.verbose)

        Weights = np.ones(S.shape)
        self.n_iter_ = []

        for _ in range(self.n_reweights):
            glasso.weights = Weights
            glasso.fit(S, mode='precomputed')

            Theta = glasso.precision_
            Weights = np.where(
                Theta == 0,
                1 / self.eps,
                np.abs(self.penalty.derivative(Theta))
            )

            if self.verbose:
                print(f"Min/Max Weights after penalty derivative: "
                      f"{Weights.min():.2e}, {Weights.max():.2e}")

            self.n_iter_.append(glasso.n_iter_)
            # TODO print losses for original problem?

        self.precision_ = glasso.precision_
        self.covariance_ = glasso.covariance_
        return self
