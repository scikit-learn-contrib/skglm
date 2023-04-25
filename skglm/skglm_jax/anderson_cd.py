from functools import partial

import jax
import jax.numpy as jnp
from skglm.skglm_jax.datafits import QuadraticJax
from skglm.skglm_jax.penalties import L1Jax


class AndersonCD:

    def __init__(self, max_iter=100, verbose=0):
        self.max_iter = max_iter
        self.verbose = verbose

    def solve(self, X, y, datafit: QuadraticJax, penalty: L1Jax):
        X, y = self._transfer_to_device(X, y)

        n_samples, n_features = X.shape
        lipschitz = datafit.get_features_lipschitz_cst(X, y)

        w = jnp.zeros(n_features)
        Xw = jnp.zeros(n_samples)
        all_features = jnp.arange(n_features)

        for it in range(self.max_iter):

            w, Xw = AndersonCD._cd_epoch(X, y, w, Xw, all_features, lipschitz,
                                         datafit, penalty)

            if self.verbose:
                p_obj = datafit.value(X, y, w) + penalty.value(w)

                grad_ws = datafit.gradient_ws(X, y, w, Xw, all_features)
                subdiff_dist = penalty.subdiff_dist(w, grad_ws, all_features)
                stop_crit = jnp.max(subdiff_dist)

                print(
                    f"Iter {it}: p_obj={p_obj:.8f} stop_crit={stop_crit:.4e}"
                )

        return w

    def _transfer_to_device(self, X, y):
        # TODO: other checks
        return jnp.asarray(X), jnp.asarray(y)

    @staticmethod
    @partial(jax.jit, static_argnums=(-2, -1))
    def _cd_epoch(X, y, w, Xw, ws, lipschitz, datafit, penalty):
        for j in ws:

            # Null columns of X would break this functions
            # as their corresponding lipschitz is 0
            # TODO: implement condition using lax
            # if lipschitz[j] == 0.:
            #     continue

            step = 1 / lipschitz[j]

            grad_j = datafit.gradient_1d(X, y, w, Xw, j)
            next_w_j = penalty.prox_1d(w[j] - step * grad_j, step)

            delta_w_j = next_w_j - w[j]

            w = w.at[j].set(next_w_j)
            Xw = Xw + delta_w_j * X[:, j]

        return w, Xw
