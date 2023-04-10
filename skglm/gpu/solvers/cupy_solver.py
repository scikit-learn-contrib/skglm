import cupy as cp
import cupyx.scipy.sparse as cpx

import numpy as np
from scipy import sparse

from skglm.gpu.solvers.base import BaseFistaSolver, BaseL1


class CupySolver(BaseFistaSolver):

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

        is_X_sparse = sparse.issparse(X)

        # transfer to device
        X_gpu = cp.array(X) if not is_X_sparse else cpx.csr_matrix(X)
        y_gpu = cp.array(y)

        # init vars in device
        w = cp.zeros(n_features)
        old_w = cp.zeros(n_features)
        mid_w = cp.zeros(n_features)
        grad = cp.zeros(n_features)

        t_old, t_new = 1, 1

        for it in range(self.max_iter):

            # compute grad
            grad = datafit.gradient(X_gpu, y_gpu, mid_w, X_gpu @ mid_w)

            # forward / backward
            w = penalty.prox(mid_w - step * grad, step)

            if self.verbose:
                w_cpu = cp.asnumpy(w)

                p_obj = datafit.value(X_gpu, y_gpu, w, X_gpu @ w) + penalty.value(w)

                grad = datafit.gradient(X, y, w_cpu, X @ w_cpu)
                opt_crit = penalty.max_subdiff_distance(w_cpu, grad)

                print(
                    f"Iteration {it:4}: p_obj={p_obj:.8f}, opt crit={opt_crit:.4e}"
                )

            # extrapolate
            mid_w = w + ((t_old - 1) / t_new) * (w - old_w)

            # update FISTA vars
            t_old = t_new
            t_new = (1 + cp.sqrt(1 + 4 * t_old ** 2)) / 2
            old_w = cp.copy(w)

        # transfer back to host
        w_cpu = cp.asnumpy(w)

        return w_cpu


class L1CuPy(BaseL1):

    def prox(self, value, stepsize):
        return cp.sign(value) * cp.maximum(cp.abs(value) - stepsize * self.alpha, 0.)
