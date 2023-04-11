# if not set, raises an error related to CUDA linking API.
# as recommended, setting the 'XLA_FLAGS' to bypass it.
# side-effect: (perhaps) slow compilation time.
import os
os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'  # noqa

import numpy as np  # noqa

import jax  # noqa
import jax.numpy as jnp  # noqa
# set float64 as default float type.
# if not, amplifies rounding errors.
jax.config.update("jax_enable_x64", True)  # noqa

from scipy import sparse  # noqa
from jax.experimental import sparse as jax_sparse  # noqa

from skglm.gpu.solvers.base import BaseFistaSolver, BaseQuadratic, BaseL1  # noqa


class JaxSolver(BaseFistaSolver):

    def __init__(self, max_iter=1000, use_auto_diff=True, verbose=0):
        self.max_iter = max_iter
        self.use_auto_diff = use_auto_diff
        self.verbose = verbose

    def solve(self, X, y, datafit, penalty):
        n_samples, n_features = X.shape

        # compute step
        lipschitz = datafit.get_lipschitz_cst(X)
        if lipschitz == 0.:
            return np.zeros(n_features)

        step = 1 / lipschitz

        # transfer to device
        if sparse.issparse(X):
            # sparse matrices are still an experimental features in jax
            # matrix operation are supported only for COO matrices but missing
            # for CSC, CSR. hence working with COO in the wait for a new Jax release
            # that adds support for these features
            X_gpu = jax_sparse.BCOO.from_scipy_sparse(X)
        else:
            X_gpu = jnp.asarray(X)
        y_gpu = jnp.asarray(y)

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

                w_cpu = np.asarray(w, dtype=np.float64)
                grad_cpu = np.asarray(grad, dtype=np.float64)
                opt_crit = penalty.max_subdiff_distance(w_cpu, grad_cpu)

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


class QuadraticJax(BaseQuadratic):

    def value(self, X_gpu, y_gpu, w):
        n_samples = X_gpu.shape[0]
        return jnp.sum((X_gpu @ w - y_gpu) ** 2) / (2. * n_samples)

    def gradient(self, X_gpu, y_gpu, w):
        n_samples = X_gpu.shape[0]
        return X_gpu.T @ (X_gpu @ w - y_gpu) / n_samples


class L1Jax(BaseL1):

    def prox(self, value, stepsize):
        return jnp.sign(value) * jnp.maximum(jnp.abs(value) - stepsize * self.alpha, 0.)
