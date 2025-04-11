# License: BSD 3 clause

from skglm.utils.data import make_dummy_covariance_data
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import pinvh

from skglm.solvers.gram_cd import barebones_cd_gram
from skglm.penalties import L0_5


class GraphicalLasso():
    """ A first-order BCD Graphical Lasso solver implementing the GLasso algorithm
    described in Friedman et al., 2008 and the P-GLasso algorithm described in
    Mazumder et al., 2012."""

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


# class AdaptiveGraphicalLasso():
#     """ An adaptive version of the Graphical Lasso that solves non-convex penalty
#     variations using the reweighting strategy from Candès et al., 2007."""

#     def __init__(
#         self,
#         alpha=1.,
#         # strategy="log",
#         n_reweights=5,
#         max_iter=1000,
#         tol=1e-8,
#         warm_start=False,
#         penalty=L0_5(1.),
#     ):
#         self.alpha = alpha
#         # self.strategy = strategy  # we can remove this param. it if not used elsewhere
#         self.n_reweights = n_reweights
#         self.max_iter = max_iter
#         self.tol = tol
#         self.warm_start = warm_start
#         self.penalty = penalty

#     def fit(self, S):
#         """ Fit the AdaptiveGraphicalLasso model on the empirical covariance matrix S."""
#         glasso = GraphicalLasso(
#             alpha=self.alpha,
#             algo="primal",
#             max_iter=self.max_iter,
#             tol=self.tol,
#             warm_start=True)
#         Weights = np.ones(S.shape)
#         self.n_iter_ = []
#         for it in range(self.n_reweights):
#             glasso.weights = Weights
#             glasso.fit(S)
#             Theta = glasso.precision_

#             Weights = abs(self.penalty.derivative(Theta))

#             self.n_iter_.append(glasso.n_iter_)
#             # TODO print losses for original problem?
#             glasso.covariance_ = np.linalg.pinv(Theta, hermitian=True)
#         self.precision_ = glasso.precision_
#         self.covariance_ = glasso.covariance_
#         return self


# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     from skglm.utils.data import make_dummy_covariance_data
#     from skglm.penalties import L1  # Import L0_5

#     # Define the dimensions for the dummy data
#     n = 100    # number of samples
#     p = 20     # number of features

#     # Create dummy covariance data
#     S, Theta_true, alpha_max = make_dummy_covariance_data(n, p)

#     # Compute the true covariance matrix as the pseudoinverse of the true precision matrix
#     true_covariance = np.linalg.pinv(Theta_true, hermitian=True)

#     # Instantiate the AdaptiveGraphicalLasso model with L0_5 penalty
#     model = AdaptiveGraphicalLasso(
#         # Pass L0_5 object
#         alpha=alpha_max * 0.1, n_reweights=5, tol=1e-8, warm_start=True, penalty=L0_5(1.))

#     # Fit the model on the empirical covariance matrix S
#     model.fit(S)

#     # Compute normalized mean squared error (NMSE) between the true and estimated covariance matrices
#     nmse = np.linalg.norm(model.covariance_ - true_covariance)**2 / \
#         np.linalg.norm(true_covariance)**2
#     print("Normalized MSE (NMSE): {:.3e}".format(nmse))

#     fig, axes = plt.subplots(1, 2, figsize=(12, 5))

#     im0 = axes[0].imshow(true_covariance, cmap="hot", interpolation="nearest")
#     axes[0].set_title("True Covariance Matrix")
#     plt.colorbar(im0, ax=axes[0])

#     im1 = axes[1].imshow(model.covariance_, cmap="hot", interpolation="nearest")
#     axes[1].set_title("Estimated Covariance Matrix\n(Adaptive Graphical Lasso)")
#     plt.colorbar(im1, ax=axes[1])

#     plt.tight_layout()
#     plt.show()

class AdaptiveGraphicalLasso():
    """ An adaptive version of the Graphical Lasso that solves non-convex penalty
    variations using the reweighting strategy from Candès et al., 2007."""

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


if __name__ == "__main__":

    # Define the dimensions for the dummy data
    n = 100    # number of samples
    p = 20     # number of features

    # Create dummy covariance data
    S, Theta_true, alpha_max = make_dummy_covariance_data(n, p)

    # Compute the true covariance matrix as the pseudoinverse of the true precision matrix
    true_covariance = np.linalg.pinv(Theta_true, hermitian=True)

    # Instantiate the AdaptiveGraphicalLasso model with L0_5 penalty
    model = AdaptiveGraphicalLasso(
        # Pass L0_5 object
        alpha=alpha_max * 0.1, n_reweights=5, tol=1e-8, warm_start=True)

    # Fit the model on the empirical covariance matrix S
    model.fit(S)

    # Compute normalized mean squared error (NMSE) between the true and estimated covariance matrices
    nmse = np.linalg.norm(model.covariance_ - true_covariance)**2 / \
        np.linalg.norm(true_covariance)**2
    print("Normalized MSE (NMSE): {:.3e}".format(nmse))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im0 = axes[0].imshow(true_covariance, cmap="hot", interpolation="nearest")
    axes[0].set_title("True Covariance Matrix")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(model.covariance_, cmap="hot", interpolation="nearest")
    axes[1].set_title("Estimated Covariance Matrix\n(Adaptive Graphical Lasso)")
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.show()
