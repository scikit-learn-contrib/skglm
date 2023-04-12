import math
import numpy as np
import numba
from numba import cuda

from skglm.gpu.solvers.base import BaseL1, BaseQuadratic, BaseFistaSolver

import warnings
from numba.core.errors import NumbaPerformanceWarning

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)


# Built from GPU properties
# Refer to utils to get GPU properties
MAX_1DIM_BLOCK = (1024,)
MAX_2DIM_BLOCK = (32, 32)
MAX_1DIM_GRID = (65535,)
MAX_2DIM_GRID = (65535, 65535)


class NumbaSolver(BaseFistaSolver):

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

        # number of block to use along features-axis when launching kernel
        n_blocks_axis_1 = math.ceil(X.shape[1] / MAX_1DIM_BLOCK[0])

        # transfer to device
        X_gpu = cuda.to_device(X)
        y_gpu = cuda.to_device(y)

        # init vars on device
        # CAUTION: should be init with specific values
        # otherwise, stale values in GPU memory are used
        w = cuda.to_device(np.zeros(n_features))
        mid_w = cuda.to_device(np.zeros(n_features))
        old_w = cuda.to_device(np.zeros(n_features))

        # needn't to be init with values as it stores results of computation
        grad = cuda.device_array(n_features)

        t_old, t_new = 1, 1

        for it in range(self.max_iter):

            # inplace update of grad
            datafit.gradient(X_gpu, y_gpu, mid_w, grad)

            # inplace update of mid_w
            _forward[n_blocks_axis_1, MAX_1DIM_BLOCK](mid_w, grad, step, mid_w)

            # inplace update of w
            penalty.prox(mid_w, step, w)

            if self.verbose:
                w_cpu = w.copy_to_host()

                p_obj = datafit.value(X, y, w_cpu, X @ w_cpu) + penalty.value(w_cpu)

                datafit.gradient(X_gpu, y_gpu, w, grad)
                grad_cpu = grad.copy_to_host()

                opt_crit = penalty.max_subdiff_distance(w_cpu, grad_cpu)

                print(
                    f"Iteration {it:4}: p_obj={p_obj:.8f}, opt crit={opt_crit:.4e}"
                )

            # extrapolate
            coef = (t_old - 1) / t_new
            # mid_w = w + coef * (w - old_w)
            _extrapolate[n_blocks_axis_1, MAX_1DIM_BLOCK](
                w, old_w, coef, mid_w)

            # update FISTA vars
            t_old = t_new
            t_new = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2
            # in `copy_to_device`: `self` is destination and `other` is source
            old_w.copy_to_device(w)

        # transfer back to host
        w_cpu = w.copy_to_host()

        return w_cpu


class QuadraticNumba(BaseQuadratic):

    def gradient(self, X_gpu, y_gpu, w, out):
        minus_residual = cuda.device_array(X_gpu.shape[0])

        n_blocks_axis_0, n_blocks_axis_1 = (math.ceil(n / MAX_1DIM_BLOCK[0])
                                            for n in X_gpu.shape)

        _compute_minus_residual[n_blocks_axis_0, MAX_1DIM_BLOCK](
            X_gpu, y_gpu, w, minus_residual)

        _compute_grad[n_blocks_axis_1, MAX_1DIM_BLOCK](X_gpu, minus_residual, out)

    def gradient_2(self, X_gpu, y_gpu, w, out):
        minus_residual = cuda.device_array(X_gpu.shape[0])

        grid_dim = tuple(math.ceil(X_gpu.shape[idx] / MAX_2DIM_BLOCK[idx])
                         for idx in range(2))

        _compute_minus_residual_2[grid_dim, MAX_2DIM_BLOCK](
            X_gpu, y_gpu, w, minus_residual)

        n_blocks_axis_1 = math.ceil(X_gpu.shape[1] / MAX_1DIM_BLOCK[0])
        _compute_grad[n_blocks_axis_1, MAX_1DIM_BLOCK](X_gpu, minus_residual, out)


@cuda.jit
def _compute_minus_residual_2(X_gpu, y_gpu, w, out):
    i, j = cuda.grid(2)

    n_samples, n_features = X_gpu.shape
    if i >= n_samples or j >= n_features:
        return

    t_i, t_j = cuda.threadIdx.x, cuda.threadIdx.y
    sub_shape = MAX_2DIM_BLOCK  # cuda.blockDim.x, cuda.blockDim.y

    sub_X = cuda.shared.array(shape=sub_shape, dtype=numba.f8)
    sub_y = cuda.shared.array(shape=sub_shape[0], dtype=numba.f8)
    sub_w = cuda.shared.array(shape=sub_shape[1], dtype=numba.f8)

    # load data in shared memory
    sub_X[t_i, t_j] = X_gpu[i, j]
    sub_y[t_i] = y_gpu[i]
    sub_w[t_j] = w[j]

    cuda.syncthreads()

    tmp = 0.
    for k in range(sub_shape[1]):
        tmp += sub_X[t_i, k] * sub_w[k]
    tmp -= sub_y[t_i]

    cuda.syncthreads()

    out[i] += tmp


class L1Numba(BaseL1):

    def prox(self, value, stepsize, out):
        level = stepsize * self.alpha

        n_blocks = math.ceil(value.shape[0] / MAX_1DIM_BLOCK[0])

        _ST_vec[n_blocks, MAX_1DIM_BLOCK](value, level, out)


@cuda.jit
def _compute_minus_residual(X_gpu, y_gpu, w, out):
    # compute: out = X_gpu @ w - y_gpu
    i = cuda.grid(1)

    n_samples, n_features = X_gpu.shape
    if i >= n_samples:
        return

    tmp = 0.
    for j in range(n_features):
        tmp += X_gpu[i, j] * w[j]
    tmp -= y_gpu[i]

    out[i] = tmp


@cuda.jit
def _compute_grad(X_gpu, minus_residual, out):
    # compute: out = X.T @ minus_residual
    j = cuda.grid(1)

    n_samples, n_features = X_gpu.shape
    if j >= n_features:
        return

    tmp = 0.
    for i in range(n_samples):
        tmp += X_gpu[i, j] * minus_residual[i] / X_gpu.shape[0]

    out[j] = tmp


@cuda.jit
def _forward(mid_w, grad, step, out):
    j = cuda.grid(1)

    n_features = len(mid_w)
    if j >= n_features:
        return

    out[j] = mid_w[j] - step * grad[j]


@cuda.jit
def _ST_vec(value, level, out):
    j = cuda.grid(1)

    n_features = value.shape[0]
    if j >= n_features:
        return

    value_j = value[j]

    if abs(value_j) <= level:
        value_j = 0.
    elif value_j > level:
        value_j = value_j - level
    else:
        value_j = value_j + level

    out[j] = value_j


@cuda.jit
def _extrapolate(w, old_w, coef, out):
    # compute: out = w + coef * (w - old_w)
    j = cuda.grid(1)

    n_features = len(w)
    if j >= n_features:
        return

    out[j] = w[j] + coef * (w[j] - old_w[j])
