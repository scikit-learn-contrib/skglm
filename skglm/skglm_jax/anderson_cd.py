from functools import partial

import jax
import numpy as np
import jax.numpy as jnp

from skglm.skglm_jax.datafits import QuadraticJax
from skglm.skglm_jax.penalties import L1Jax
from skglm.skglm_jax.utils import JaxAA


class AndersonCD:

    EPS_TOL = 0.3

    def __init__(self, max_iter=100, max_epochs=100, tol=1e-6, p0=10,
                 use_acc=False, verbose=0):
        self.max_iter = max_iter
        self.max_epochs = max_epochs
        self.tol = tol
        self.p0 = p0
        self.use_acc = use_acc
        self.verbose = verbose

    def solve(self, X, y, datafit: QuadraticJax, penalty: L1Jax):
        X, y = self._transfer_to_device(X, y)

        n_samples, n_features = X.shape
        lipschitz = datafit.get_features_lipschitz_cst(X, y)

        w = jnp.zeros(n_features)
        Xw = jnp.zeros(n_samples)
        all_features = jnp.full(n_features, fill_value=True, dtype=bool)

        for it in range(self.max_iter):

            # check convergence
            grad = datafit.gradient_ws(X, y, w, Xw, all_features)
            scores = penalty.subdiff_dist_ws(w, grad, all_features)
            stop_crit = jnp.max(scores)

            if self.verbose:
                p_obj = datafit.value(X, y, w) + penalty.value(w)

                print(
                    f"Iteration {it}: p_obj_in={p_obj:.8f} "
                    f"stop_crit_in={stop_crit:.4e}"
                )

            if stop_crit <= self.tol:
                break

            # build ws
            gsupp_size = penalty.generalized_support(w).sum()
            ws_size = min(
                max(2 * gsupp_size, self.p0),
                n_features
            )

            ws = jnp.full(n_features, fill_value=False, dtype=bool)
            ws_features = jnp.argsort(scores)[-ws_size:]
            ws = ws.at[ws_features].set(True)

            tol_in = AndersonCD.EPS_TOL * stop_crit

            w, Xw = self._solve_sub_problem(X, y, w, Xw, ws, lipschitz, tol_in,
                                            datafit, penalty)

        w_cpu = np.asarray(w)
        return w_cpu

    def _solve_sub_problem(self, X, y, w, Xw, ws, lipschitz, tol_in,
                           datafit, penalty):

        if self.use_acc:
            accelerator = JaxAA(K=5)

        for epoch in range(self.max_epochs):

            w, Xw = self._cd_epoch(X, y, w, Xw, ws, lipschitz,
                                   datafit, penalty)

            if self.use_acc:
                w, Xw = accelerator.extrapolate(w, Xw)

            # check convergence
            grad_ws = datafit.gradient_ws(X, y, w, Xw, ws)
            scores_ws = penalty.subdiff_dist_ws(w, grad_ws, ws)
            stop_crit_in = jnp.max(scores_ws)

            if max(self.verbose - 1, 0):
                p_obj_in = datafit.value(X, y, w) + penalty.value(w)

                print(
                    f"Epoch {epoch}: p_obj_in={p_obj_in:.8f} "
                    f"stop_crit_in={stop_crit_in:.4e}"
                )

            if stop_crit_in <= tol_in:
                break

        return w, Xw

    @partial(jax.jit, static_argnums=(0, -2, -1))
    def _cd_epoch(self, X, y, w, Xw, ws, lipschitz, datafit, penalty):
        for j, in_ws in enumerate(ws):

            w, Xw = jax.lax.cond(
                in_ws,
                lambda X, y, w, Xw, j, lipschitz: self._cd_epoch_j(X, y, w, Xw, j, lipschitz, datafit, penalty),  # noqa
                lambda X, y, w, Xw, j, lipschitz: (w, Xw),
                *(X, y, w, Xw, j, lipschitz)
            )

        return w, Xw

    @partial(jax.jit, static_argnums=(0, -2, -1))
    def _cd_epoch_j(self, X, y, w, Xw, j, lipschitz, datafit, penalty):

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

    def _transfer_to_device(self, X, y):
        # TODO: other checks
        # - skip if they are already jax array
        return jnp.asarray(X), jnp.asarray(y)
