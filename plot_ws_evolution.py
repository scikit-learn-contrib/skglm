import numpy as np
from numpy.linalg import norm
from numba import njit

from skglm.datafits.base import BaseMultitaskDatafit
from skglm.penalties.base import BasePenalty

from skglm.penalties.block_separable import SparseGroupL1
from skglm.datafits.multi_task import QuadraticGroup

from skglm.utils import make_correlated_data, grp_converter

import matplotlib.pyplot as plt


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
    ws_sizes = []
    Xw = X @ w  # this would make sense when w is provided
    datafit.initialize(X, y)

    for k in range(1, max_iter+1):
        ws = np.arange(n_groups)
        is_optimal, scores = _check_is_optimal(datafit, penalty, y, X, w, Xw,
                                               ws, stop_tol)

        if is_optimal:
            print("Outer Solver: early exit")
            break

        ws, ws_size = _select_ws(w, n_groups, penalty, scores, p0)
        ws_sizes.append(ws_size)

        if verbose == 1:
            p_obj = datafit.value(y, w, Xw) + penalty.value(w)
            print(f"Iteration {k}: {p_obj}, support: {ws_size} / {n_groups}")

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
    return w, ws_sizes


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
        grad.append(-datafit.gradient_j(X, y, w, Xw, g))

    scores = penalty.subdiff_distance(w, grad, ws)
    is_optimal = np.max(scores) < stop_tol

    return is_optimal, scores


@njit
def _select_ws(w, n_groups, penalty, scores, p0):
    size_support = penalty.generalized_support(w).sum()

    ws_size = max(p0,
                  min(n_groups, int(1.5 * size_support)))

    ws = np.zeros(ws_size, np.int32)
    for k in range(ws_size):
        top_kth_grp = np.argmax(scores)

        ws[k] = top_kth_grp
        scores[top_kth_grp] = -np.inf

    return ws, ws_size


if __name__ == '__main__':
    n_samples, n_features = 100, 1000
    X, y, _ = make_correlated_data(n_samples, n_features, random_state=42)

    groups = 2  # contiguous groups of 10 elements
    n_groups = n_features // groups

    grp_ptr, grp_indices = grp_converter(groups, n_features)
    weights = np.ones(n_groups)  # abs(np.random.randn(n_groups))

    weighted_XTy = X.T@y / np.repeat(weights, groups)
    alpha_max = np.max(
        norm(weighted_XTy.reshape(-1, groups), ord=2, axis=1)) / n_samples

    quad_group = QuadraticGroup(grp_ptr, grp_indices)
    group_penalty = SparseGroupL1(
        alpha=1., tau=0., grp_ptr=grp_ptr, grp_indices=grp_indices, weights=weights)

    p0 = 10
    ratios = np.array([1/2**i for i in range(0, 10, 2)])
    dict_ws_sizes = {}
    for ratio in ratios:
        group_penalty.alpha = ratio * alpha_max
        _, ws_sizes = group_solver(X, y, quad_group, group_penalty,
                                   max_iter=1000, max_epochs=100, p0=p0)

        dict_ws_sizes[ratio] = ws_sizes

    for i, (ratio, ws_sizes) in enumerate(dict_ws_sizes.items()):
        plt.plot(ws_sizes, '-o', label=r"$\alpha/\alpha_{max}=%.4f$" % ratio)

    plt.axhline(y=p0, color='gray', linestyle='dashed')

    plt.legend()
    plt.xlabel("iteration")
    plt.ylabel("working set size")
    plt.show()
