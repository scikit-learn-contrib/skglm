import math
import numpy as np
from numba import cuda


from skglm.gpu.utils.host_utils import compute_obj, eval_opt_crit


# max number of threads that one block could contain
# https://forums.developer.nvidia.com/t/maximum-number-of-threads-on-thread-block/46392
N_THREADS = 1024


class NumbaSolver:

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

        # number of block to use along each axis when calling kernel
        n_blocks_axis_0, n_blocks_axis_1 = [math.ceil(n / N_THREADS) for n in X.shape]

        # transfer to device
        X_gpu = cuda.as_cuda_array(X)
        y_gpu = cuda.as_cuda_array(y)

        # init vars on device
        w = cuda.device_array(n_features)
        old_w = cuda.device_array(n_features)
        mid_w = cuda.device_array(n_features)

        grad = cuda.device_array(n_features)
        minus_residual = cuda.device_array(n_samples)

        t_old, t_new = 1, 1

        for it in range(self.max_iter):

            __compute_minus_residual[n_blocks_axis_0, N_THREADS](
                X_gpu, y_gpu, w, out=minus_residual)

            __compute_grad[n_blocks_axis_1, N_THREADS](
                X_gpu, minus_residual, out=grad)

            __forward_backward[n_blocks_axis_1, N_THREADS](
                mid_w, grad, step, lmbd, out=w)

            if self.verbose:
                p_obj = compute_obj(X, y, lmbd, w)
                opt_crit = eval_opt_crit(X, y, lmbd, w)

                print(
                    f"Iteration {it:4}: p_obj={p_obj:.8f}, opt crit={opt_crit:.4e}"
                )

            # extrapolate
            coef = (t_old - 1) / t_new
            __extrapolate[n_blocks_axis_1, N_THREADS](w, old_w, coef, out=mid_w)

            # update FISTA vars
            t_old = t_new
            t_new = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2
            # in `copy_to_device`: `self` is destination and `other` is source
            old_w.copy_to_device(w)

        return w


def __compute_minus_residual(X_gpu, y_gpu, w, out):
    # X_gpu @ w - y_gpu
    pass


def __compute_grad(X_gpu, minus_residual, out):
    # X.T @ minus_residual
    pass


def __forward_backward():
    # mid_w = mid_w - step * grad
    # w = ST_vec(mid_w, step * lmbd)
    pass


def __extrapolate(w, old_w, coef, out):
    # mid_w = w + ((t_old - 1) / t_new) * (w - old_w)
    pass
