import jax
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
        n_features = X.shape[1]
        Xw_minus_y = Xw - y

        grad_ws = jnp.empty(n_features)
        for j, in_ws in enumerate(ws):

            grad_j = jax.lax.cond(
                in_ws,
                lambda X, Xw_minus_y, j: X[:, j] @ Xw_minus_y / len(Xw_minus_y),
                lambda X, Xw_minus_y, j: 0.,
                *(X, Xw_minus_y, j)
            )

            grad_ws = grad_ws.at[j].set(grad_j)

        return grad_ws

    def get_features_lipschitz_cst(self, X, y):
        n_samples = X.shape[0]
        return jnorm(X, ord=2, axis=0) ** 2 / n_samples
