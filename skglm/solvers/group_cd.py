import numpy as np
from numpy.linalg import norm
from numba import njit

from skglm.datafits.base import BaseMultitaskDatafit
from skglm.penalties.base import BasePenalty

from skglm.penalties.block_separable import SparseGroupL1
from skglm.datafits.multi_task import QuadraticGroup

from skglm.utils import make_correlated_data, grp_converter


def group_solver(X: np.ndarray, y: np.ndarray,
                 datafit: BaseMultitaskDatafit, penalty: BasePenalty,
                 max_iter: int = 1000, stop_tol=1e-7,
                 verbose: bool = False) -> np.ndarray:

    n_features = X.shape[1]
    w = np.zeros(n_features, dtype=np.float64)
    grp_ptr, grp_indices = penalty.grp_ptr, penalty.grp_indices

    prev_obj: float = 0.
    current_obj: float = 0.

    # init
    Xw = X @ w  # this would make sense when w is provided
    datafit.initialize(X, y)

    for k in range(1, max_iter+1):
        prev_obj, current_obj = _cycle_group_cd(datafit, penalty,
                                                w, Xw, grp_ptr, grp_indices)
        # naive stopping criterion
        if abs(current_obj - prev_obj) < stop_tol:
            break

        if verbose:
            print(f"Iteration {k}: {current_obj}")

    return w


@njit
def _cycle_group_cd(datafit, penalty, w, Xw, grp_ptr, grp_indices):
    n_groups = len(grp_ptr) - 1

    prev_obj = datafit.value(y, w, Xw) + penalty.value(w)

    for g in range(n_groups):
        grp_g_indices = grp_indices[grp_ptr[g]:grp_ptr[g+1]]
        X_g = X[:, grp_g_indices]
        old_w_g = w[grp_g_indices].copy()

        inv_lipschitz_g = 1. / datafit.lipschitz[g]
        grad_g_datafit = datafit.gradient_j(X, y, w, Xw, g)

        w[grp_g_indices] = penalty.prox_1feat(
            old_w_g - inv_lipschitz_g * grad_g_datafit,
            inv_lipschitz_g, g)

        Xw += X_g @ (w[grp_g_indices] - old_w_g)

    current_obj = datafit.value(y, w, Xw) + penalty.value(w)

    return prev_obj, current_obj


if __name__ == '__main__':
    n_samples, n_features = 100, 1000
    X, y, _ = make_correlated_data(n_samples, n_features)
    alpha_max = norm(X.T@y, ord=np.inf) / n_samples

    grp_ptr, grp_indices = grp_converter(1, n_features)
    weights = np.array([1 for _ in range(n_features)], dtype=np.float64)
    alpha = alpha_max / 10.

    quad_group = QuadraticGroup(grp_ptr, grp_indices)
    group_penalty = SparseGroupL1(
        alpha, tau=0., grp_ptr=grp_ptr, grp_indices=grp_indices, weights=weights)

    w = group_solver(X, y, quad_group, group_penalty, max_iter=1000, verbose=True)
    pass
