import numpy as np
from numpy.linalg import norm
from numba import njit

from skglm.datafits.base import BaseMultitaskDatafit
from skglm.penalties.base import BasePenalty

from skglm.penalties.block_separable import SparseGroupL1
from skglm.datafits.multi_task import QuadraticGroup

from skglm.utils import make_correlated_data, grp_converter


FREQ_CHECK = 10


def group_solver(X: np.ndarray, y: np.ndarray,
                 datafit: BaseMultitaskDatafit, penalty: BasePenalty,
                 max_iter: int = 1000, max_epochs: int = 100,
                 stop_tol: float = 1e-7, p0: int = 10,
                 verbose: bool = False) -> np.ndarray:

    n_features: int = X.shape[1]
    grp_ptr, grp_indices = penalty.grp_ptr, penalty.grp_indices
    n_groups: int = len(grp_ptr) - 1
    ws_size: int = 0

    # init
    w = np.zeros(n_features, dtype=np.float64)
    Xw = X @ w  # this would make sense when w is provided
    datafit.initialize(X, y)

    for k in range(1, max_iter+1):
        ws = np.arange(n_groups)
        is_optimal, scores = _check_is_optimal(datafit, penalty, y, X, w, Xw,
                                               ws, stop_tol)
        if is_optimal:
            print("Outer Solver: early exit")
            break

        ws_size = max(p0,
                      min(n_features, 2 * penalty.generalized_support(w)))
        ws = np.argpartition(scores, -ws_size)[-ws_size:]

        if verbose == 1:
            p_obj = datafit.value(y, w, Xw) + penalty.value(w)
            print(f"Iteration {k}: {p_obj}")

        for epoch in np.arange(1, max_epochs+1):
            # inplace update
            _cycle_group_cd(datafit, penalty, y, X,
                            w, Xw, grp_ptr, grp_indices, ws)

            if verbose == 2:
                p_obj = datafit.value(y, w, Xw) + penalty.value(w)
                print(f"Epoch {epoch}: {p_obj}")

            if epoch % FREQ_CHECK == 0:
                is_optimal, scores = _check_is_optimal(datafit, penalty, y, X,
                                                       w, Xw, ws, stop_tol)
                if is_optimal:
                    print("Inner Solver: early exit")
                    break
    return w


@njit
def _cycle_group_cd(datafit, penalty, y, X, w, Xw, grp_ptr, grp_indices, ws):

    for g in range(ws):
        grp_g_indices = grp_indices[grp_ptr[g]:grp_ptr[g+1]]
        X_g = X[:, grp_g_indices]
        old_w_g = w[grp_g_indices].copy()

        inv_lipschitz_g = 1. / datafit.lipschitz[g]
        grad_g_datafit = datafit.gradient_j(X, y, w, Xw, g)

        w[grp_g_indices] = penalty.prox_1feat(
            old_w_g - inv_lipschitz_g * grad_g_datafit,
            inv_lipschitz_g, g)

        if w[grp_g_indices] != old_w_g:
            Xw += X_g @ (w[grp_g_indices] - old_w_g)


@njit
def _check_is_optimal(datafit, penalty, y, X, w, Xw, ws, stop_tol):
    grad = np.vstack((datafit.gradient_j(X, y, w, Xw, g)
                      for g in ws))
    scores = penalty.subdiff_distance(w, grad, ws)
    is_optimal = np.max(scores) < stop_tol

    return is_optimal, scores


if __name__ == '__main__':
    pass
