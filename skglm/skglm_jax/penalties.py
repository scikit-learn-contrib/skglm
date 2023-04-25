import jax.numpy as jnp


class L1Jax:

    def __init__(self, alpha):
        self.alpha = alpha

    def value(self, w):
        return (self.alpha * jnp.abs(w)).sum()

    def prox_1d(self, value, stepsize):
        shifted_value = jnp.abs(value) - stepsize * self.alpha
        return jnp.sign(value) * jnp.maximum(shifted_value, 0.)

    def subdiff_dist(self, w, grad, ws):
        dist = jnp.zeros(len(ws))

        for idx, j in enumerate(ws):
            w_j = w[j]
            grad_j = grad[j]

            if w_j == 0.:
                dist_j = max(abs(grad_j) - self.alpha, 0.)
            else:
                dist_j = abs(grad_j + jnp.sign(w_j) * self.alpha)

            dist[idx] = dist_j

        return dist
