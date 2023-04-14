import numpy as np
from skglm.gpu.solvers.base import BaseFistaSolver


class CPUSolver(BaseFistaSolver):

    def __init__(self, max_iter=1000, verbose=0):
        self.max_iter = max_iter
        self.verbose = verbose

    def solve(self, X, y, datafit, penalty):
        n_samples, n_features = X.shape

        # compute step
        lipschitz = datafit.get_lipschitz_cst(X)
        if lipschitz == 0.:
            return np.zeros(n_features)

        step = 1 / lipschitz

        # init vars
        w = np.zeros(n_features)
        old_w = np.zeros(n_features)
        mid_w = np.zeros(n_features)
        grad = np.zeros(n_features)

        t_old, t_new = 1, 1

        for it in range(self.max_iter):

            # compute grad
            grad = datafit.gradient(X, y, mid_w, X @ mid_w)

            # forward / backward
            w = penalty.prox(mid_w - step * grad, step)

            if self.verbose:
                p_obj = datafit.value(X, y, w, X @ w) + penalty.value(w)

                grad = datafit.gradient(X, y, w, X @ w)
                opt_crit = penalty.max_subdiff_distance(w, grad)

                print(
                    f"Iteration {it:4}: p_obj={p_obj:.8f}, opt crit={opt_crit:.4e}"
                )

            # extrapolate
            mid_w = w + ((t_old - 1) / t_new) * (w - old_w)

            # update FISTA vars
            t_old = t_new
            t_new = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2
            old_w = np.copy(w)

        return w
