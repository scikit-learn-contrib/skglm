# License: BSD 3 clause

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from skglm.penalties.separable import LogSumPenalty
from sklearn.datasets import make_sparse_spd_matrix
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import pinvh

from skglm.solvers.gram_cd import barebones_cd_gram
from skglm.penalties import L0_5


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


class AdaptiveGraphicalLassoPenalty():
    """An adaptive version of the Graphical Lasso with non-convex penalties.

    Solves non-convex penalty variations using the reweighting strategy
    from Candès et al., 2007.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization parameter controlling sparsity.
    n_reweights : int, default=5
        Number of reweighting iterations.
    max_iter : int, default=1000
        Maximum iterations for inner solver.
    tol : float, default=1e-8
        Convergence tolerance.
    warm_start : bool, default=False
        Whether to use warm start.
    penalty : Penalty object, default=L0_5(1.)
        Non-convex penalty function to use for reweighting.
    """

    def __init__(
        self,
        alpha=1.,
        # strategy="log",
        n_reweights=5,
        max_iter=1000,
        tol=1e-8,
        warm_start=False,
        penalty=L0_5(1.),
    ):
        self.alpha = alpha
        # self.strategy = strategy  # we can remove this param. it if not used elsewhere
        self.n_reweights = n_reweights
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.penalty = penalty

    def fit(self, S):
        """Fit the AdaptiveGraphicalLasso model on the empirical covariance matrix S."""
        glasso = GraphicalLasso(
            alpha=self.alpha,
            algo="primal",
            max_iter=self.max_iter,
            tol=self.tol,
            warm_start=True)
        Weights = np.ones(S.shape)
        self.n_iter_ = []
        for it in range(self.n_reweights):
            glasso.weights = Weights
            glasso.fit(S)
            Theta = glasso.precision_

            Theta_sym = (Theta + Theta.T) / 2
            Weights = np.where(
                Theta_sym == 0,
                1 / self.penalty.eps,
                np.abs(self.penalty.derivative(Theta_sym))
            )

            print(
                f"Min/Max Weights after penalty derivative: "
                f"{Weights.min():.2e}, {Weights.max():.2e}")

            self.n_iter_.append(glasso.n_iter_)
            # TODO print losses for original problem?

            glasso.covariance_ = np.linalg.pinv(Theta, hermitian=True)
        self.precision_ = glasso.precision_
        self.covariance_ = glasso.covariance_
        if not np.isclose(self.alpha, self.penalty.alpha):
            print(
                f"Alpha mismatch: GLasso alpha = {self.alpha}, "
                f"Penalty alpha = {self.penalty.alpha}")
        else:
            print(f"Alpha values match: {self.alpha}")
        return self

# TODO: remove this class and use AdaptiveGraphicalLassoPenalty instead


class AdaptiveGraphicalLasso():
    """An adaptive version of the Graphical Lasso with non-convex penalties.

    Solves non-convex penalty variations using the reweighting strategy
    from Candès et al., 2007.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization parameter controlling sparsity.
    strategy : str, default="log"
        Reweighting strategy: "log", "sqrt", or "mcp".
    n_reweights : int, default=5
        Number of reweighting iterations.
    max_iter : int, default=1000
        Maximum iterations for inner solver.
    tol : float, default=1e-8
        Convergence tolerance.
    warm_start : bool, default=False
        Whether to use warm start.
    """

    def __init__(
        self,
        alpha=1.,
        strategy="log",
        n_reweights=5,
        max_iter=1000,
        tol=1e-8,
        warm_start=False,
    ):
        self.alpha = alpha
        self.strategy = strategy
        self.n_reweights = n_reweights
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start

    def fit(self, S):
        glasso = GraphicalLasso(
            alpha=self.alpha,
            algo="primal",
            max_iter=self.max_iter,
            tol=self.tol,
            warm_start=True)
        Weights = np.ones(S.shape)
        self.n_iter_ = []
        for it in range(self.n_reweights):
            glasso.weights = Weights
            glasso.fit(S)
            Theta = glasso.precision_
            Weights = update_weights(Theta, self.alpha, strategy=self.strategy)
            self.n_iter_.append(glasso.n_iter_)
            # TODO print losses for original problem?
            glasso.covariance_ = np.linalg.pinv(Theta, hermitian=True)
        self.precision_ = glasso.precision_
        self.covariance_ = glasso.covariance_
        return self


def update_weights(Theta, alpha, strategy="log"):
    """Update weights for adaptive graphical lasso based on strategy.

    Parameters
    ----------
    Theta : array-like
        Precision matrix.
    alpha : float
        Regularization parameter.
    strategy : str, default="log"
        Reweighting strategy: "log", "sqrt", or "mcp".

    Returns
    -------
    array-like
        Updated weights.
    """
    if strategy == "log":
        return 1/(np.abs(Theta) + 1e-10)
    elif strategy == "sqrt":
        return 1/(2*np.sqrt(np.abs(Theta)) + 1e-10)
    elif strategy == "mcp":
        gamma = 3.
        Weights = np.zeros_like(Theta)
        Weights[np.abs(Theta)
                < gamma*alpha] = (alpha -
                                  np.abs(Theta[np.abs(Theta)
                                               < gamma*alpha])/gamma)
        return Weights
    else:
        raise ValueError(f"Unknown strategy {strategy}")

# TODO: remove this testing code
# Testing


def _frobenius_norm_diff(A, B):
    return np.linalg.norm(A - B, ord='fro') / np.linalg.norm(B, ord='fro')


def _generate_problem(dim=20, n_samples=100, seed=42):
    np.random.seed(seed)

    # Ground-truth sparse precision matrix (positive definite)
    Theta_true = make_sparse_spd_matrix(n_dim=dim, alpha=0.95, smallest_coef=0.1)
    Sigma_true = np.linalg.inv(Theta_true)

    # Sample from multivariate normal to get empirical covariance matrix S
    X = np.random.multivariate_normal(np.zeros(dim), Sigma_true, size=n_samples)
    S = np.cov(X, rowvar=False)

    return S, Theta_true


if __name__ == "__main__":
    # Set test parameters
    dim = 20
    alpha = 0.1
    n_reweights = 5
    seed = 42

    # Get empirical covariance and ground truth
    S, Theta_true = _generate_problem(dim=dim, seed=seed)

    # Define non-convex penalty — this is consistent with 'log' strategy
    penalty = LogSumPenalty(alpha=alpha, eps=1e-10)

    # Fit new penalty-based model
    model_penalty = AdaptiveGraphicalLassoPenalty(
        alpha=alpha,
        penalty=penalty,
        n_reweights=n_reweights,
    )
    model_penalty.fit(S)

    # Fit old strategy-based model
    model_strategy = AdaptiveGraphicalLasso(
        alpha=alpha,
        strategy="log",
        n_reweights=n_reweights,
    )
    model_strategy.fit(S)

    # Extract precision matrices
    Theta_penalty = model_penalty.precision_
    Theta_strategy = model_strategy.precision_

    # Compare the two estimated models
    rel_diff_between_models = _frobenius_norm_diff(Theta_penalty, Theta_strategy)
    print(
        f"\n Frobenius norm relative difference between models: "
        f"{rel_diff_between_models:.2e}")
    print(" Matrices are close?", np.allclose(
        Theta_penalty, Theta_strategy, atol=1e-4))

    # Compare both to ground truth
    rel_diff_penalty_vs_true = _frobenius_norm_diff(Theta_penalty, Theta_true)
    rel_diff_strategy_vs_true = _frobenius_norm_diff(Theta_strategy, Theta_true)

    print(
        f"\n Penalty vs true Θ:   Frobenius norm diff = {rel_diff_penalty_vs_true:.2e}")
    print(
        f"Strategy vs true Θ:  Frobenius norm diff = {rel_diff_strategy_vs_true:.2e}")

    print("\nTrue precision matrix:\n", Theta_true)
    print("\nPenalty-based estimate:\n", Theta_penalty)
    print("\nStrategy-based estimate:\n", Theta_strategy)

    # Visualization
    n_features = Theta_true.shape[0]

    plt.close('all')
    cmap = plt.cm.bwr

    matrices = [Theta_true, Theta_penalty, Theta_strategy]
    titles = [r"$\Theta_{\mathrm{True}}$",
              r"$\Theta_{\mathrm{Penalty}}$", r"$\Theta_{\mathrm{Strategy}}$"]

    fig, ax = plt.subplots(3, 1, layout="constrained",
                           figsize=(4.42, 9.33))

    vmax = max(np.max(mat) for mat in matrices) / 2
    vmin = min(np.min(mat) for mat in matrices) / 2
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0)

    for i in range(3):
        im = ax[i].imshow(matrices[i], cmap=cmap, norm=norm)
        sparsity = 100 * (1 - np.count_nonzero(matrices[i]) / (n_features**2))
        ax[i].set_title(f"{titles[i]}\nsparsity = {sparsity:.2f}%", fontsize=12)
        ax[i].set_xticks([])
        ax[i].set_yticks([])

    divider = make_axes_locatable(ax[i])
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    ticks_loc = cbar.ax.get_yticks().tolist()
    cbar.ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    cbar.ax.set_yticklabels([f'{i:.0e}' for i in cbar.get_ticks()])
    cbar.ax.tick_params(labelsize=10)

    plt.show()
