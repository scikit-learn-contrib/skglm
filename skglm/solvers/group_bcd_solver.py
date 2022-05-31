import numpy as np
from numba import njit


def bcd_solver(X, y, datafit, penalty, w_init=None,
               max_iter=1000, tol=1e-7, verbose=False):
    """Run a group BCD solver.

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        Design matrix.

    y : array, shape (n_samples,)
        Target vector.

    datafit : QuadraticGroup
        DataFit object.

    penalty : WeightedGroupL1
        Penalty object.

    w_init : array, shape (n_features,), default None
        Initial value of coefficients.
        If set to None, a zero vector is used instead.

    max_iter : int, default 1000
        Maximum number of iterations.

    tol : float, default 1e-7
        Tolerance for convergence.

    verbose : bool, default False
        Log or not the objective at each iteration.

    Returns
    -------
    w : array, shape (n_features,)
        Solution that minimizes the problem defined by datafit and penalty.
    """
    n_features = X.shape[1]
    n_groups = len(penalty.grp_ptr) - 1

    # init
    w = w_init or np.zeros(n_features)
    Xw = X @ w
    datafit.initialize(X, y)
    all_groups = np.arange(n_groups)

    for k in range(max_iter):
        prev_p_obj = datafit.value(y, w, Xw) + penalty.value(w)
        _bcd_epoch(X, y, w, Xw, datafit, penalty, all_groups)
        current_p_obj = datafit.value(y, w, Xw) + penalty.value(w)

        if verbose:
            print(f"Iteration {k}: {current_p_obj}")

        if np.abs(current_p_obj - prev_p_obj) <= tol:  # naive stopping criterion
            print("Early exit")
            break

    return w


@njit
def _bcd_epoch(X, y, w, Xw, datafit, penalty, ws):
    """Perform a single BCD epoch on groups in ws."""
    grp_ptr, grp_indices = penalty.grp_ptr, penalty.grp_indices

    for g in ws:
        grp_g_indices = grp_indices[grp_ptr[g]: grp_ptr[g+1]]
        old_w_g = w[grp_g_indices].copy()

        inv_lipschitz_g = 1 / datafit.lipschitz[g]
        grad_g = datafit.gradient_g(X, y, w, Xw, g)

        w[grp_g_indices] = penalty.prox_1group(
            old_w_g - inv_lipschitz_g * grad_g,
            inv_lipschitz_g, g
        )

        # update Xw without copying w_g
        for i, g_i in enumerate(grp_g_indices):
            if old_w_g[i] != w[g_i]:
                Xw += (w[g_i] - old_w_g[i]) * X[:, g_i]
    return
