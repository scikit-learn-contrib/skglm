import jax.numpy as jnp
from jax.numpy.linalg import norm as jnorm

from skglm.skglm_jax.utils import jax_jit_method


class QuadraticJax:
    """1 / (2 n_samples) ||y - Xw||^2"""

    def value(self, X, y, w):
        n_samples = X.shape[0]
        return ((X @ w - y) ** 2).sum() / (2. * n_samples)

    def gradient_1d(self, X, y, w, Xw, j):
        n_samples = X.shape[0]
        return X[:, j] @ (Xw - y) / n_samples

    @jax_jit_method
    def gradient_ws(self, X, y, w, Xw, ws):
        n_samples = X.shape[0]
        Xw_minus_y = Xw - y

        grad_ws = jnp.zeros(len(ws))
        for idx, j in enumerate(ws):
            grad_j = X[:, j] @ Xw_minus_y / n_samples
            grad_ws = grad_ws.at[idx].set(grad_j)

        return grad_ws

    def get_features_lipschitz_cst(self, X, y):
        n_samples = X.shape[0]
        return jnorm(X, ord=2, axis=0) ** 2 / n_samples
