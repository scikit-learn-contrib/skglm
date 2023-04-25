from functools import partial

import jax
import jax.numpy as jnp


class L1Jax:
    """alpha ||w||_1"""

    def __init__(self, alpha):
        self.alpha = alpha

    def value(self, w):
        return (self.alpha * jnp.abs(w)).sum()

    def prox_1d(self, value, stepsize):
        shifted_value = jnp.abs(value) - stepsize * self.alpha
        return jnp.sign(value) * jnp.maximum(shifted_value, 0.)

    @partial(jax.jit, static_argnums=(0,))
    def subdiff_dist(self, w, grad, ws):
        dist = jnp.zeros(len(ws))

        for idx, j in enumerate(ws):
            w_j = w[j]
            grad_j = grad[j]

            dist_j = jax.lax.cond(
                w_j == 0.,
                lambda w_j, grad_j, alpha: jnp.maximum(jnp.abs(grad_j) - alpha, 0.),
                lambda w_j, grad_j, alpha: jnp.abs(grad_j + jnp.sign(w_j) * alpha),
                w_j, grad_j, self.alpha
            )

            dist = dist.at[idx].set(dist_j)

        return dist
