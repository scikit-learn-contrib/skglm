# License: BSD 3 clause

from skglm.solvers.gram_cd import barebones_cd_gram
from scipy.linalg import pinvh
import numpy as np
from skglm.penalties.separable import L0_5


class GraphicalLasso():
    """A first-order BCD Graphical Lasso solver.

    Implementing the GLasso algorithm described in Friedman et al., 2008 and
    the P-GLasso algorithm described in Mazumder et al., 2012.
    """

    def __init__(self,
                 alpha=1.,
                 weights=None,
                 algo="dual",
                 max_iter=100,
                 tol=1e-8,
                 warm_start=False,
                 inner_tol=1e-4,
                 verbose=False
                 ):
        self.alpha = alpha
        self.weights = weights
        self.algo = algo
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.inner_tol = inner_tol
        self.verbose = verbose

    def fit(self, S):
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
            W = S.copy()
            W *= 0.95
            diagonal = S.flat[:: p + 1]
            W.flat[:: p + 1] = diagonal
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


class AdaptiveGraphicalLasso():
    """An adaptive version of the Graphical Lasso with non-convex penalties.

    Solves non-convex penalty variations using the reweighting strategy
    from Candès et al., 2007.

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

    def fit(self, S):
        """Fit the AdaptiveGraphicalLasso model on the empirical covariance matrix S."""
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
            glasso.fit(S)

            Theta_sym = (glasso.precision_ + glasso.precision_.T) / 2
            Weights = np.where(
                Theta_sym == 0,
                1 / self.eps,
                np.abs(self.penalty.derivative(Theta_sym))
            )

            if self.verbose:
                print(f"Min/Max Weights after penalty derivative: "
                      f"{Weights.min():.2e}, {Weights.max():.2e}")

            self.n_iter_.append(glasso.n_iter_)
            # TODO print losses for original problem?

        self.precision_ = glasso.precision_
        self.covariance_ = glasso.covariance_
        return self
