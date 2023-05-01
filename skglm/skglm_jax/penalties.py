import jax
import jax.numpy as jnp

from skglm.skglm_jax.utils import jax_jit_method


class L1Jax:
    """alpha ||w||_1"""

    def __init__(self, alpha):
        self.alpha = alpha

    def value(self, w):
        return (self.alpha * jnp.abs(w)).sum()

    def prox_1d(self, value, stepsize):
        shifted_value = jnp.abs(value) - stepsize * self.alpha
        return jnp.sign(value) * jnp.maximum(shifted_value, 0.)

    @jax_jit_method
    def subdiff_dist_ws(self, w, grad_ws, ws):
        n_features = w.shape[0]
        dist = jnp.empty(n_features)

        for j, in_ws in enumerate(ws):
            w_j = w[j]
            grad_j = grad_ws[j]

            dist_j = jax.lax.cond(
                in_ws,
                self._compute_subdiff_dist_j,
                lambda w_j, grad_j: 0.,
                *(w_j, grad_j)
            )

            dist = dist.at[j].set(dist_j)

        return dist

    def generalized_support(self, w):
        return w != 0.

    @jax_jit_method
    def _compute_subdiff_dist_j(self, w_j, grad_j):
        dist_j = jax.lax.cond(
            w_j == 0.,
            lambda w_j, grad_j, alpha: jnp.maximum(jnp.abs(grad_j) - alpha, 0.),
            lambda w_j, grad_j, alpha: jnp.abs(grad_j + jnp.sign(w_j) * alpha),
            *(w_j, grad_j, self.alpha)
        )
        return dist_j
