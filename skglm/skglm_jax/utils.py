import jax
import jax.numpy as jnp
from functools import partial


jax_jit_method = partial(jax.jit, static_argnums=(0,))


class JaxAA:

    def __init__(self, K):
        self.K, self.current_iter = K, 0
        self.arr_w_, self.arr_Xw_ = None, None

    def extrapolate(self, w, Xw):
        if self.arr_w_ is None or self.arr_Xw_ is None:
            self.arr_w_ = jnp.zeros((w.shape[0], self.K+1))
            self.arr_Xw_ = jnp.zeros((Xw.shape[0], self.K+1))

        if self.current_iter <= self.K:
            self.arr_w_ = self.arr_w_.at[:, self.current_iter].set(w)
            self.arr_Xw_ = self.arr_Xw_.at[:, self.current_iter].set(Xw)
            self.current_iter += 1
            return w, Xw

        # compute residuals
        U = jnp.diff(self.arr_w_, axis=1)

        # compute extrapolation coefs
        try:
            inv_UTU_ones = jnp.linalg.solve(U.T @ U, jnp.ones(self.K))
        except Exception:
            return w, Xw
        finally:
            self.current_iter = 0

        # extrapolate
        C = inv_UTU_ones / jnp.sum(inv_UTU_ones)

        return self.arr_w_[:, 1:] @ C, self.arr_Xw_[:, 1:] @ C
