# if not set, raises an error related to CUDA linking API.
# as recommended, setting the 'XLA_FLAGS' to bypass it.
# side-effect: (perhaps) slow compilation time.
import os
os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'  # noqa

import numpy as np

import jax
import jax.numpy as jnp
# set float64 as default float type.
# if not, amplifies rounding errors.
jax.config.update("jax_enable_x64", True)  # noqa

from skglm.gpu.utils.host_utils import compute_obj, eval_opt_crit


class JaxSolver:

    def __init__(self, max_iter=1000, use_auto_diff=True, verbose=0):
        self.max_iter = max_iter
        self.use_auto_diff = use_auto_diff
        self.verbose = verbose

    def solve(self, X, y, lmbd):
        n_samples, n_features = X.shape

        # compute step
        lipschitz = np.linalg.norm(X, ord=2) ** 2
        if lipschitz == 0.:
            return np.zeros(n_features)

        step = 1 / lipschitz

        # transfer to device
        X_gpu = jnp.asarray(X)
        y_gpu = jnp.asarray(y)

        # get grad func of datafit
        if self.use_auto_diff:
            grad_quad_loss = jax.grad(_quad_loss)

        # init vars in device
        w = jnp.zeros(n_features)
        old_w = jnp.zeros(n_features)
        mid_w = jnp.zeros(n_features)
        grad = jnp.zeros(n_features)

        t_old, t_new = 1, 1

        for it in range(self.max_iter):

            # compute grad
            if self.use_auto_diff:
                grad = grad_quad_loss(mid_w, X_gpu, y_gpu)
            else:
                grad = jnp.dot(X_gpu.T, jnp.dot(X_gpu, mid_w) - y_gpu)

            # forward / backward
            mid_w = mid_w - step * grad
            w = jnp.sign(mid_w) * jnp.maximum(jnp.abs(mid_w) - step * lmbd, 0.)

            if self.verbose:
                w_cpu = np.asarray(w, dtype=np.float64)

                p_obj = compute_obj(X, y, lmbd, w_cpu)
                opt_crit = eval_opt_crit(X, y, lmbd, w_cpu)

                print(
                    f"Iteration {it:4}: p_obj={p_obj:.8f}, opt crit={opt_crit:.4e}"
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


def _quad_loss(w, X_gpu, y_gpu):
    pred_y = jnp.dot(X_gpu, w)
    return 0.5 * jnp.sum((y_gpu - pred_y) ** 2)
