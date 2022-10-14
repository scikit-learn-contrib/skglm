import numpy as np
from numba import njit
from skglm.solvers.base import BaseSolver
from skglm.solvers.common import construct_grad


@njit
def _prox_vec(w, z, penalty, lipschitz):
    n_features = w.shape[0]
    for j in range(n_features):
        w[j] = penalty.prox_1d(z[j], 1 / lipschitz, j)
    return w


class FISTA(BaseSolver):
    r"""ISTA solver with Nesterov acceleration (FISTA)."""

    def __init__(self, max_iter=100, tol=1e-4, fit_intercept=False, warm_start=False,
                 opt_freq=10, verbose=0):
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.warm_start = warm_start
        self.opt_freq = opt_freq
        self.verbose = verbose

    def solve(self, X, y, datafit, penalty, w_init=None, Xw_init=None):
        n_samples, n_features = X.shape
        all_features = np.arange(n_features)
        t_new = 1

        w = w_init.copy() if w_init is not None else np.zeros(n_features)
        z = w_init.copy() if w_init is not None else np.zeros(n_features)
        Xw = Xw_init.copy() if Xw_init is not None else np.zeros(n_samples)

        # TODO: OR line search
        lipschitz = datafit.global_lipschitz

        for n_iter in range(self.max_iter):
            t_old = t_new
            t_new = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2
            w_old = w.copy()
            grad = construct_grad(X, y, z, X @ z, datafit, all_features)
            z -= grad / lipschitz
            w = _prox_vec(w, z, penalty, lipschitz)
            Xw = X @ w
            z = w + (t_old - 1.) / t_new * (w - w_old)

            if n_iter % self.opt_freq == 0:
                opt = penalty.subdiff_distance(w, grad, all_features)
                stop_crit = np.max(opt)

                if self.verbose:
                    p_obj = datafit.value(y, w, Xw) + penalty.value(w)
                    print(
                        f"Iteration {n_iter+1}: {p_obj:.10f}, "
                        f"stopping crit: {stop_crit:.2e}"
                    )

                if stop_crit < self.tol:
                    if self.verbose:
                        print(f"Stopping criterion max violation: {stop_crit:.2e}")
                    break
        return w
