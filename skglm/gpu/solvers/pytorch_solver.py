import torch

import numpy as np
from scipy import sparse

from skglm.gpu.solvers.base import BaseFistaSolver, BaseQuadratic, BaseL1


class PytorchSolver(BaseFistaSolver):

    def __init__(self, max_iter=1000, use_auto_diff=True, verbose=0):
        self.max_iter = max_iter
        self.use_auto_diff = use_auto_diff
        self.verbose = verbose

    def solve(self, X, y, datafit, penalty):
        n_samples, n_features = X.shape
        X_is_sparse = sparse.issparse(X)

        # compute step
        lipschitz = datafit.get_lipschitz_cst(X)
        if lipschitz == 0.:
            return np.zeros(n_features)

        step = 1 / lipschitz

        # transfer data
        selected_device = torch.device("cuda")
        if X_is_sparse:
            X_gpu = torch.sparse_csc_tensor(
                X.indptr, X.indices, X.data, X.shape,
                dtype=torch.float64,
                device=selected_device
            )
        else:
            X_gpu = torch.tensor(X, device=selected_device)
        y_gpu = torch.tensor(y, device=selected_device)

        # init vars
        w = torch.zeros(n_features, dtype=torch.float64, device=selected_device)
        old_w = torch.zeros(n_features, dtype=torch.float64, device=selected_device)
        mid_w = torch.zeros(n_features, dtype=torch.float64, device=selected_device,
                            requires_grad=self.use_auto_diff)

        t_old, t_new = 1, 1

        for it in range(self.max_iter):

            # compute gradient
            if self.use_auto_diff:
                datafit_value = datafit.value(X_gpu, y_gpu, mid_w)
                datafit_value.backward()

                grad = mid_w.grad
            else:
                grad = datafit.gradient(X_gpu, y_gpu, mid_w)

            # forward / backward
            with torch.no_grad():
                w = penalty.prox(mid_w - step * grad, step)

            if self.verbose:
                # transfer back to host
                w_cpu = w.cpu().numpy()

                p_obj = datafit.value(X, y, w_cpu) + penalty.value(w_cpu)

                if self.use_auto_diff:
                    w_tmp = torch.tensor(w, dtype=torch.float64,
                                         device=selected_device, requires_grad=True)

                    datafit_value = datafit.value(X_gpu, y_gpu, w_tmp)
                    datafit_value.backward()

                    grad_cpu = w_tmp.grad.detach().cpu().numpy()
                else:
                    grad_cpu = datafit.gradient(X, y, w_cpu)

                opt_crit = penalty.max_subdiff_distance(w_cpu, grad_cpu)

                print(
                    f"Iteration {it:4}: p_obj={p_obj:.8f}, opt crit={opt_crit:.4e}"
                )

            # extrapolate
            mid_w = w + ((t_old - 1) / t_new) * (w - old_w)
            mid_w = mid_w.requires_grad_(self.use_auto_diff)

            # update FISTA vars
            t_old = t_new
            t_new = 0.5 * (1 + np.sqrt(1. + 4. * t_old ** 2))
            # no need to copy `w` since its update (forward/backward)
            # creates a new instance
            old_w = w

        # transfer back to host
        w_cpu = w.cpu().numpy()

        return w_cpu


class QuadraticPytorch(BaseQuadratic):

    def value(self, X, y, w):
        return ((y - X @ w) ** 2).sum() / (2 * len(y))

    def gradient(self, X, y, w):
        return X.T @ (X @ w - y) / X.shape[0]


class L1Pytorch(BaseL1):

    def prox(self, value, stepsize):
        shifted_value = torch.abs(value) - stepsize * self.alpha
        return torch.sign(value) * torch.maximum(shifted_value, torch.tensor(0.))
