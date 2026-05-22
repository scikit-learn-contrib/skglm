import numpy as np

import jax
import jax.numpy as jnp

from skglm.skglm_jax.datafits import QuadraticJax
from skglm.skglm_jax.penalties import L1Jax


class Fista:

    def __init__(self, max_iter=200, use_auto_diff=True, verbose=0):
        self.max_iter = max_iter
        self.use_auto_diff = use_auto_diff
        self.verbose = verbose

    def solve(self, X, y, datafit: QuadraticJax, penalty: L1Jax):
        n_samples, n_features = X.shape
        X_gpu, y_gpu = jnp.asarray(X), jnp.asarray(y)

        # compute step
        lipschitz = datafit.get_global_lipschitz_cst(X_gpu, y_gpu)
        if lipschitz == 0.:
            return np.zeros(n_features)

        step = 1 / lipschitz
        all_features = jnp.full(n_features, fill_value=True, dtype=bool)

        # get grad func of datafit
        if self.use_auto_diff:
            auto_grad = jax.jit(jax.grad(datafit.value, argnums=-1))

        # init vars in device
        w = jnp.zeros(n_features)
        old_w = jnp.zeros(n_features)
        mid_w = jnp.zeros(n_features)
        grad = jnp.zeros(n_features)

        t_old, t_new = 1, 1

        for it in range(self.max_iter):

            # compute grad
            if self.use_auto_diff:
                grad = auto_grad(X_gpu, y_gpu, mid_w)
            else:
                grad = datafit.gradient(X_gpu, y_gpu, mid_w)

            # forward / backward
            val = mid_w - step * grad
            w = penalty.prox(val, step)

            if self.verbose:
                p_obj = datafit.value(X_gpu, y_gpu, w) + penalty.value(w)

                if self.use_auto_diff:
                    grad = auto_grad(X_gpu, y_gpu, w)
                else:
                    grad = datafit.gradient(X_gpu, y_gpu, w)

                scores = penalty.subdiff_dist_ws(w, grad, all_features)
                stop_crit = jnp.max(scores)

                print(
                    f"Iteration {it:4}: p_obj={p_obj:.8f}, opt crit={stop_crit:.4e}"
                )

            # extrapolate
            mid_w = w + ((t_old - 1) / t_new) * (w - old_w)

            # update FISTA vars
            t_old = t_new
            t_new = 0.5 * (1 + jnp.sqrt(1. + 4. * t_old ** 2))
            old_w = jnp.copy(w)

        # transfer back to host
        w_cpu = np.asarray(w, dtype=np.float64)

        return w_cpu
