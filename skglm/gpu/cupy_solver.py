import cupy as cp
import numpy as np
from numpy.linalg import norm

from skglm.gpu.utils.host_utils import compute_obj, eval_opt_crit


class CupySolver:

    def __init__(self, max_iter=1000, verbose=0):
        self.max_iter = max_iter
        self.verbose = verbose

    def solve(self, X, y, lmbd):
        n_samples, n_features = X.shape

        # compute step
        lipschitz = np.linalg.norm(X, ord=2) ** 2
        if lipschitz == 0.:
            return np.zeros(n_features)

        step = 1 / lipschitz

        # transfer to device
        X_gpu = cp.array(X)
        y_gpu = cp.array(y)

        # init vars in device
        w = cp.zeros(n_features)
        old_w = cp.zeros(n_features)
        mid_w = cp.zeros(n_features)
        grad = cp.zeros(n_features)

        t_old, t_new = 1, 1

        for it in range(self.max_iter):

            # compute grad
            cp.dot(X_gpu.T, X_gpu @ mid_w - y_gpu, out=grad)

            # forward / backward: w = ST(mid_w - step * grad, step * lmbd)
            mid_w = mid_w - step * grad
            w = cp.sign(mid_w) * cp.maximum(cp.abs(mid_w) - step * lmbd, 0.)

            if self.verbose:
                w_cpu = cp.asnumpy(w)

                p_obj = compute_obj(X, y, lmbd, w_cpu)
                opt_crit = eval_opt_crit(X, y, lmbd, w_cpu)

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
