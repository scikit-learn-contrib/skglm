import numpy as np
from numba import njit

from skglm.datafits.base import BaseMultitaskDatafit
from skglm.penalties.base import BasePenalty

from skglm.penalties.block_separable import SparseGroupL1
from skglm.datafits.multi_task import QuadraticGroup


@njit
def group_solver(X: np.ndarray, y: np.ndarray,
                 datafit: BaseMultitaskDatafit, penalty: BasePenalty,
                 max_iter: int = 1000, verbose: bool = False):

    _, n_features = X.shape
    w = np.zeros(n_features, dtype=np.float64)
    grp_ptr, grp_indices = penalty.grp_ptr, penalty.grp_indices
    n_groups = len(grp_ptr) - 1

    # init
    Xw = X @ w
    datafit.initialize(X, y)

    for k in range(max_iter):
        for g in range(n_groups):
            grp_g_indices = grp_indices[grp_ptr[g]:grp_ptr[g+1]]

            lipschitz_g = datafit.lipschitz[g]
            grad_g_datafit = datafit.gradient_j(X, y, w, Xw, g)

            w[grp_g_indices] = penalty.prox_1feat(
                w[grp_g_indices] - grad_g_datafit / lipschitz_g,
                1. / lipschitz_g, g)

        if verbose:
            objective = datafit.value(y, w, Xw) + penalty.value(w)
            print(f"Iteration {k}: {objective}")

    return w
