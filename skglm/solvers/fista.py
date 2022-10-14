import numpy as np
from numba import njit
from skglm.solvers.common import dist_fix_point
from skglm.solvers.base import BaseSolver


def dist_fix_point(w, grad_ws, lipschitz, penalty, ws):
    dist_fix_point = np.zeros(ws.shape[0])
    dist_fix_point = np.abs(w - penalty.prox_vec(w - grad_ws / lipschitz, 1 / lipschitz))
    return dist_fix_point


class FISTA(BaseSolver):
    r"""ISTA solver with Nesterov acceleration (FISTA)."""

    def __init__(self, max_iter=100, tol=1e-4, fit_intercept=False, warm_start=False,
                 opt_freq=50, verbose=0):
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.warm_start = warm_start
        self.opt_freq = opt_freq
        self.verbose = verbose

    def solve(self, X, y, penalty, w_init=None, weights=None):
        # needs a quadratic datafit, but works with L1, WeightedL1, SLOPE
        n_samples, n_features = X.shape
        all_features = np.arange(n_features)
        t_new = 1

        w = w_init.copy() if w_init is not None else np.zeros(n_features)
        z = w_init.copy() if w_init is not None else np.zeros(n_features)
        weights = weights if weights is not None else np.ones(n_features)

        # FISTA with Gram update
        G = X.T @ X
        Xty = X.T @ y
        lipschitz = np.linalg.norm(X, ord=2) ** 2 / n_samples

        for n_iter in range(self.max_iter):
            t_old = t_new
            t_new = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2
            w_old = w.copy()
            grad = (G @ z - Xty) / n_samples
            z -= grad / lipschitz
            # TODO: TO DISCUSS!
            # XXX: should add a full prox update
            # for j in range(n_features):
            #     w[j] = penalty.prox_1d(z[j], 1 / lipschitz, j)
            w = penalty.prox_vec(z, 1 / lipschitz)
            z = w + (t_old - 1.) / t_new * (w - w_old)

            if n_iter % self.opt_freq == 0:
                # opt = penalty.subdiff_distance(w, grad, all_features)
                opt = dist_fix_point(w, grad, lipschitz, penalty, all_features) 
                stop_crit = np.max(opt)

                if self.verbose:
                    p_obj = (np.sum((y - X @ w) ** 2) / (2 * n_samples)
                             + penalty.value(w))
                    print(
                        f"Iteration {n_iter+1}: {p_obj:.10f}, "
                        f"stopping crit: {stop_crit:.2e}"
                    )

                if stop_crit < self.tol:
                    if self.verbose:
                        print(f"Stopping criterion max violation: {stop_crit:.2e}")
                    break
        return w
