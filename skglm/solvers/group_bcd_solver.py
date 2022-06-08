import numpy as np
from numba import njit


def bcd_solver(X, y, datafit, penalty, w_init=None, p0=2,
               max_iter=1000, max_epochs=100, tol=1e-7, verbose=False):
    """Run a group BCD solver.

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        Design matrix.

    y : array, shape (n_samples,)
        Target vector.

    datafit : instance of BaseDatafit
        Datafit object.

    penalty : instance of BasePenalty
        Penalty object.

    w_init : array, shape (n_features,), default None
        Initial value of coefficients.
        If set to None, a zero vector is used instead.

    p0 : int, default 2
        Min number of groups to be included in the working set.

    max_iter : int, default 1000
        Maximum number of iterations.

    max_epochs : int, default 100
        Maximum number of epochs.

    tol : float, default 1e-6
        Tolerance for convergence.

    verbose : bool, default False
        Amount of verbosity. 0/False is silent.

    Returns
    -------
    w : array, shape (n_features,)
        Solution that minimizes the problem defined by datafit and penalty.

    p_objs_out: array (max_iter,)
        The objective values at every outer iteration.

    stop_crit: float
        The value of the stop criterion.
    """
    n_features = X.shape[1]
    n_groups = len(penalty.grp_ptr) - 1

    # init
    w = np.zeros(n_features) if w_init is None else w_init
    Xw = X @ w
    datafit.initialize(X, y)
    all_groups = np.arange(n_groups)
    p_objs_out = np.zeros(max_iter)
    stop_crit = 0  # prevent ref before assignment when max_iter == 0

    for t in range(max_iter):
        if t == 0:  # avoid computing grad and opt twice
            grad = _construct_grad(X, y, w, Xw, datafit, all_groups)
            opt = penalty.subdiff_distance(w, grad, all_groups)
            stop_crit = np.max(opt)

            if stop_crit <= tol:
                print("Outer solver: Early exit")
                break

        gsupp_size = penalty.generalized_support(w).sum()
        ws_size = max(min(p0, n_groups),
                      min(n_groups, 2 * gsupp_size))
        ws = np.argpartition(opt, -ws_size)[-ws_size:]  # k-largest items [no sort]

        for epoch in range(max_epochs):
            _bcd_epoch(X, y, w, Xw, datafit, penalty, ws)

            if epoch % 10 == 0:
                grad_ws = _construct_grad(X, y, w, Xw, datafit, ws)
                opt_in = penalty.subdiff_distance(w, grad_ws, ws)
                stop_crit_in = np.max(opt_in)

                if max(verbose - 1, 0):
                    current_p_obj = datafit.value(y, w, Xw) + penalty.value(w)
                    print(
                        f"Epoch {epoch+1}: {current_p_obj:.10f} "
                        f"obj. variation: {stop_crit_in:.2e}"
                    )

                if stop_crit_in <= 0.3 * stop_crit:
                    print("Early exit")
                    break

        current_p_obj = datafit.value(y, w, Xw) + penalty.value(w)
        grad = _construct_grad(X, y, w, Xw, datafit, all_groups)
        opt = penalty.subdiff_distance(w, grad, all_groups)
        stop_crit = np.max(opt)

        if max(verbose, 0):
            print(
                f"Iteration {t+1}: {current_p_obj:.10f}, "
                f"stopping crit: {stop_crit:.2e}"
            )

        if stop_crit <= tol:
            print("Outer solver: Early exit")
            break

        p_objs_out[t] = current_p_obj

    return w, p_objs_out, stop_crit


@njit
def _bcd_epoch(X, y, w, Xw, datafit, penalty, ws):
    """Perform a single BCD epoch on groups in ``ws``."""
    grp_ptr, grp_indices = penalty.grp_ptr, penalty.grp_indices

    for g in ws:
        grp_g_indices = grp_indices[grp_ptr[g]: grp_ptr[g+1]]
        old_w_g = w[grp_g_indices].copy()

        lipschitz_g = datafit.lipschitz[g]
        grad_g = datafit.gradient_g(X, y, w, Xw, g)

        w[grp_g_indices] = penalty.prox_1group(
            old_w_g - grad_g / lipschitz_g,
            1 / lipschitz_g, g
        )

        for idx, j in enumerate(grp_g_indices):
            if old_w_g[idx] != w[j]:
                Xw += (w[j] - old_w_g[idx]) * X[:, j]
    return


def _construct_grad(X, y, w, Xw, datafit, ws):
    """Compute the -gradient according to each group in ``ws``.

    Note: -gradients are stacked in a 1d array (e.g. [-grad_g1, -grad_g2, ...]).
    """
    grads = np.array([])
    for g in ws:
        grad_g = -datafit.gradient_g(X, y, w, Xw, g)
        grads = np.concatenate((grads, grad_g))
    return grads
