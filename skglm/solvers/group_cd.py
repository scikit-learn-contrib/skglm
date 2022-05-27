import numpy as np
from numpy.linalg import norm
from numba import njit, jit

from skglm.datafits.base import BaseMultitaskDatafit
from skglm.penalties.base import BasePenalty

from skglm.penalties.block_separable import SparseGroupL1
from skglm.datafits.multi_task import QuadraticGroup

from skglm.utils import make_correlated_data, grp_converter
from celer import GroupLasso


FREQ_CHECK = 10


def group_solver(X: np.ndarray, y: np.ndarray,
                 datafit: BaseMultitaskDatafit, penalty: BasePenalty,
                 max_iter: int = 1000, max_epochs: int = 100,
                 stop_tol: float = 1e-7, p0: int = 10,
                 verbose: bool = False) -> np.ndarray:

    n_features: int = X.shape[1]
    grp_ptr, grp_indices = penalty.grp_ptr, penalty.grp_indices
    n_groups: int = len(grp_ptr) - 1

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

        ws, _ = _select_ws(w, penalty, scores, p0)

        if verbose == 1:
            p_obj = datafit.value(y, w, Xw) + penalty.value(w)
            print(f"Iteration {k}: {p_obj}")

        for epoch in range(1, max_epochs+1):
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

    for g in ws:
        grp_g_indices = grp_indices[grp_ptr[g]:grp_ptr[g+1]]
        X_g = X[:, grp_g_indices]
        old_w_g = w[grp_g_indices].copy()

        inv_lipschitz_g = 1. / datafit.lipschitz[g]
        grad_g_datafit = datafit.gradient_j(X, y, w, Xw, g)

        w[grp_g_indices] = penalty.prox_1feat(
            old_w_g - inv_lipschitz_g * grad_g_datafit,
            inv_lipschitz_g, g)

        if norm(w[grp_g_indices] - old_w_g) != 0:
            Xw += X_g @ (w[grp_g_indices] - old_w_g)


@njit
def _check_is_optimal(datafit, penalty, y, X, w, Xw, ws, stop_tol):
    grad = []
    for g in ws:
        grad.append(datafit.gradient_j(X, y, w, Xw, g))

    scores = penalty.subdiff_distance(w, grad, ws)
    is_optimal = np.max(scores) < stop_tol

    return is_optimal, scores


@njit
def _select_ws(w, penalty, scores, p0):
    n_features = w.shape[0]
    size_support = penalty.generalized_support(w).sum()

    ws_size = max(p0,
                  min(n_features, 2 * size_support))

    ws = np.zeros(ws_size, np.int32)
    for i in range(ws_size):
        ws[i] = np.argmax(scores)
        np.delete(scores, ws[i])

    return ws, ws_size


if __name__ == '__main__':
    n_samples, n_features = 1000, 100
    groups = 100  # contiguous groups of 100 features
    X, y, _ = make_correlated_data(n_samples, n_features, random_state=42)
    grp_ptr, grp_indices = grp_converter(groups, n_features)

    n_groups = n_features // groups
    weights = np.ones(n_groups)

    alpha_max = norm(X.T @ y / np.repeat(weights, groups), ord=np.inf) / n_samples
    alpha = alpha_max / 10.

    group_penalty = SparseGroupL1(
        alpha=alpha, tau=0., weights=weights, grp_ptr=grp_ptr, grp_indices=grp_indices)

    group_datafit = QuadraticGroup(grp_ptr, grp_indices)

    group_solver(X, y, group_datafit, group_penalty, verbose=1)
    pass
