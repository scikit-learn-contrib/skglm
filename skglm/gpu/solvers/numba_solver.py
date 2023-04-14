import math
import numpy as np
from numba import cuda

from scipy import sparse

from skglm.gpu.solvers.base import BaseL1, BaseQuadratic, BaseFistaSolver

import warnings
from numba.core.errors import NumbaPerformanceWarning

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)


# Built from GPU properties
# Refer to `utils` to get GPU properties
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
        X_is_sparse = sparse.issparse(X)

        # compute step
        lipschitz = datafit.get_lipschitz_cst(X)
        if lipschitz == 0.:
            return np.zeros(n_features)

        step = 1 / lipschitz

        # number of block to use along features-axis when launching kernel
        n_blocks_axis_1 = math.ceil(X.shape[1] / MAX_1DIM_BLOCK[0])

        # transfer to device
        if X_is_sparse:
            X_gpu_bundles = (
                cuda.to_device(X.data),
                cuda.to_device(X.indptr),
                cuda.to_device(X.indices),
                X.shape,
            )
        else:
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
            if X_is_sparse:
                datafit.sparse_gradient(*X_gpu_bundles, y_gpu, mid_w, grad)
            else:
                datafit.gradient(X_gpu, y_gpu, mid_w, grad)

            # inplace update of mid_w
            _forward[n_blocks_axis_1, MAX_1DIM_BLOCK](mid_w, grad, step, mid_w)

            # inplace update of w
            penalty.prox(mid_w, step, w)

            if self.verbose:
                w_cpu = w.copy_to_host()

                p_obj = datafit.value(X, y, w_cpu, X @ w_cpu) + penalty.value(w_cpu)

                if X_is_sparse:
                    datafit.sparse_gradient(*X_gpu_bundles, y_gpu, w, grad)
                else:
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

        QuadraticNumba._compute_minus_residual[n_blocks_axis_0, MAX_1DIM_BLOCK](
            X_gpu, y_gpu, w, minus_residual)

        QuadraticNumba._compute_grad[n_blocks_axis_1, MAX_1DIM_BLOCK](
            X_gpu, minus_residual, out)

    def sparse_gradient(self, X_gpu_data, X_gpu_indptr, X_gpu_indices, X_gpu_shape,
                        y_gpu, w, out):
        minus_residual = cuda.to_device(np.zeros(X_gpu_shape[0]))

        n_blocks = math.ceil(X_gpu_shape[1] / MAX_1DIM_BLOCK[0])

        QuadraticNumba._sparse_compute_minus_residual[n_blocks, MAX_1DIM_BLOCK](
            X_gpu_data, X_gpu_indptr, X_gpu_indices, X_gpu_shape,
            y_gpu, w, minus_residual)

        QuadraticNumba._sparse_compute_grad[n_blocks, MAX_1DIM_BLOCK](
            X_gpu_data, X_gpu_indptr, X_gpu_indices, X_gpu_shape,
            minus_residual, out)

    @staticmethod
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

    @staticmethod
    @cuda.jit
    def _compute_grad(X_gpu, minus_residual, out):
        # compute: out = X.T @ minus_residual
        j = cuda.grid(1)

        n_samples, n_features = X_gpu.shape
        if j >= n_features:
            return

        tmp = 0.
        for i in range(n_samples):
            tmp += X_gpu[i, j] * minus_residual[i] / n_samples

        out[j] = tmp

    @staticmethod
    @cuda.jit
    def _sparse_compute_minus_residual(X_gpu_data, X_gpu_indptr, X_gpu_indices,
                                       X_gpu_shape, y_gpu, w, out):
        j = cuda.grid(1)

        n_samples, n_features = X_gpu_shape
        stride_y = cuda.gridDim.x * cuda.blockDim.x

        for jj in range(j, n_features, stride_y):

            # out -= y_gpu
            # small hack to perform this operation using
            # the (features) threads instead of launching others
            for idx in range(jj, n_samples, n_features):
                cuda.atomic.sub(out, idx, y_gpu[idx])

            for idx in range(X_gpu_indptr[jj], X_gpu_indptr[jj+1]):
                i = X_gpu_indices[idx]
                cuda.atomic.add(out, i, w[jj] * X_gpu_data[idx])

    @staticmethod
    @cuda.jit
    def _sparse_compute_grad(X_gpu_data, X_gpu_indptr, X_gpu_indices, X_gpu_shape,
                             minus_residual, out):
        j = cuda.grid(1)

        n_samples, n_features = X_gpu_shape
        stride_y = cuda.gridDim.x * cuda.blockDim.x

        for jj in range(j, n_features, stride_y):
            tmp = 0.
            for idx in range(X_gpu_indptr[jj], X_gpu_indptr[jj+1]):
                i = X_gpu_indices[idx]
                tmp += X_gpu_data[idx] * minus_residual[i] / n_samples

            out[jj] = tmp


class L1Numba(BaseL1):

    def prox(self, value, stepsize, out):
        level = stepsize * self.alpha

        n_blocks = math.ceil(value.shape[0] / MAX_1DIM_BLOCK[0])

        L1Numba._ST_vec[n_blocks, MAX_1DIM_BLOCK](value, level, out)

    @staticmethod
    @cuda.jit
    def _ST_vec(value, level, out):
        j = cuda.grid(1)

        n_features = value.shape[0]
        stride_y = cuda.gridDim.x * cuda.blockDim.x

        for jj in range(j, n_features, stride_y):
            value_j = value[jj]

            if abs(value_j) <= level:
                value_j = 0.
            elif value_j > level:
                value_j = value_j - level
            else:
                value_j = value_j + level

            out[jj] = value_j


# solver kernels
@cuda.jit
def _forward(mid_w, grad, step, out):
    j = cuda.grid(1)

    n_features = mid_w.shape[0]
    stride_y = cuda.gridDim.x * cuda.blockDim.x

    for jj in range(j, n_features, stride_y):
        out[jj] = mid_w[jj] - step * grad[jj]


@cuda.jit
def _extrapolate(w, old_w, coef, out):
    # compute: out = w + coef * (w - old_w)
    j = cuda.grid(1)

    n_features = w.shape[0]
    stride_y = cuda.gridDim.x * cuda.blockDim.x

    for jj in range(j, n_features, stride_y):
        out[jj] = w[jj] + coef * (w[jj] - old_w[jj])
