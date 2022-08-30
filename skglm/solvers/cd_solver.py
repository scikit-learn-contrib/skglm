import numpy as np
from numba import njit
from scipy import sparse
from sklearn.utils import check_array
from skglm.solvers.common import construct_grad, construct_grad_sparse, dist_fix_point

from skglm.utils import AndersonAcceleration


def cd_solver_path(X, y, datafit, penalty, alphas=None, fit_intercept=False,
                   coef_init=None, max_iter=20, max_epochs=50_000,
                   p0=10, tol=1e-4, return_n_iter=False,
                   ws_strategy="subdiff", verbose=0):
    r"""Compute optimization path with Anderson accelerated coordinate descent.

    The loss is customized by passing various choices of datafit and penalty:
    loss = datafit.value() + penalty.value()

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        Training data.

    y : array, shape (n_samples,)
        Target values.

    datafit : instance of Datafit
        Datafitting term.

    penalty : instance of Penalty
        Penalty used in the model.

    alphas : ndarray
        List of alphas where to compute the models.

    fit_intercept : bool
        Whether or not to fit an intercept.

    coef_init : ndarray, shape (n_features + 1,) | None, optional, (default=None)
        Initial value of coefficients. If None, np.zeros(n_features) is used.

    max_iter : int, optional
        The maximum number of iterations (definition of working set and
        resolution of problem restricted to features in working set).

    max_epochs : int, optional
        Maximum number of (block) CD epochs on each subproblem.

    p0 : int, optional
        First working set size.

    tol : float, optional
        The tolerance for the optimization.

    return_n_iter : bool, optional
        If True, number of iterations along the path are returned.

    ws_strategy : ('subdiff'|'fixpoint'), optional
        The score used to build the working set.

    verbose : bool or int, optional
        Amount of verbosity. 0/False is silent.

    Returns
    -------
    alphas : array, shape (n_alphas,)
        The alphas along the path where models are computed.

    coefs : array, shape (n_features + 1, n_alphas)
        Coefficients along the path.

    stop_crit : array, shape (n_alphas,)
        Value of stopping criterion at convergence along the path.

    n_iters : array, shape (n_alphas,), optional
        The number of iterations along the path.
    """
    X = check_array(X, 'csc', dtype=[np.float64, np.float32],
                    order='F', copy=False, accept_large_sparse=False)
    y = check_array(y, 'csc', dtype=X.dtype.type, order='F', copy=False,
                    ensure_2d=False)

    if sparse.issparse(X):
        datafit.initialize_sparse(X.data, X.indptr, X.indices, y)
    else:
        datafit.initialize(X, y)
    n_features = X.shape[1]
    if alphas is None:
        raise ValueError('alphas should be passed explicitly')
        # if hasattr(penalty, "alpha_max"):
        #     if sparse.issparse(X):
        #         grad0 = construct_grad_sparse(
        #             X.data,  X.indptr, X.indices, y, np.zeros(n_features), len(y),
        #             datafit, np.arange(n_features))
        #     else:
        #         grad0 = construct_grad(
        #             X, y, np.zeros(n_features), len(y),
        #             datafit, np.arange(n_features))

        #     alpha_max = penalty.alpha_max(grad0)
        #     alphas = alpha_max * np.geomspace(1, eps, n_alphas, dtype=X.dtype)
        # else:
    # else:
        # alphas = np.sort(alphas)[::-1]

    n_alphas = len(alphas)
    coefs = np.zeros((n_features + fit_intercept, n_alphas), order='F', dtype=X.dtype)
    stop_crits = np.zeros(n_alphas)

    if return_n_iter:
        n_iters = np.zeros(n_alphas, dtype=int)

    for t in range(n_alphas):
        alpha = alphas[t]
        penalty.alpha = alpha
        if verbose:
            to_print = "##### Computing alpha %d/%d" % (t + 1, n_alphas)
            print("#" * len(to_print))
            print(to_print)
            print("#" * len(to_print))
        if t > 0:
            w = coefs[:, t - 1].copy()
            # TODO tmp fix debug for L05:  p0 > replace by 1 (?)
            p0 = max(np.sum(penalty.generalized_support(w)), p0)
        else:
            if coef_init is not None:
                w = coef_init.copy()
                supp_size = penalty.generalized_support(w[:n_features]).sum()
                p0 = max(supp_size, p0)
                if supp_size:
                    Xw = X @ w[:n_features] + fit_intercept * w[-1]
                # TODO explain/clean this hack
                else:
                    Xw = np.zeros_like(y)
            else:
                w = np.zeros(n_features + fit_intercept, dtype=X.dtype)
                Xw = np.zeros(X.shape[0], dtype=X.dtype)

        sol = cd_solver(
            X, y, datafit, penalty, w, Xw, fit_intercept=fit_intercept,
            max_iter=max_iter, max_epochs=max_epochs, p0=p0, tol=tol,
            verbose=verbose, ws_strategy=ws_strategy)

        coefs[:, t] = sol[0]
        stop_crits[t] = sol[-1]

        if return_n_iter:
            n_iters[t] = len(sol[1])

    results = alphas, coefs, stop_crits
    if return_n_iter:
        results += (n_iters,)
    return results


def cd_solver(
        X, y, datafit, penalty, w, Xw, fit_intercept=True, max_iter=50,
        max_epochs=50_000, p0=10, tol=1e-4, ws_strategy="subdiff",
        verbose=0):
    r"""Run a coordinate descent solver.

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        Training data.

    y : array, shape (n_samples,)
        Target values.

    datafit : instance of Datafit class
        Datafitting term.

    penalty : instance of Penalty class
        Penalty used in the model.

    w : array, shape (n_features,)
        Coefficient vector.

    Xw : array, shape (n_samples,)
        Model fit.

    fit_intercept : bool
        Whether or not to fit an intercept.

    max_iter : int, optional
        The maximum number of iterations (definition of working set and
        resolution of problem restricted to features in working set).

    max_epochs : int, optional
        Maximum number of (block) CD epochs on each subproblem.

    p0 : int, optional
        First working set size.

    tol : float, optional
        The tolerance for the optimization.

    ws_strategy : ('subdiff'|'fixpoint'), optional
        The score used to build the working set.

    verbose : bool or int, optional
        Amount of verbosity. 0/False is silent.

    Returns
    -------
    coefs : array, shape (n_features + fit_intercept, n_alphas)
        Coefficients along the path.

    obj_out : array, shape (n_iter,)
        The objective values at every outer iteration.

    stop_crit : float
        Value of stopping criterion at convergence.
    """
    if ws_strategy not in ("subdiff", "fixpoint"):
        raise ValueError(f'Unsupported value for ws_strategy: {ws_strategy}')
    n_samples, n_features = X.shape
    pen = penalty.is_penalized(n_features)
    unpen = ~pen
    n_unpen = unpen.sum()
    obj_out = []
    all_feats = np.arange(n_features)
    stop_crit = np.inf  # initialize for case n_iter=0
    w_acc, Xw_acc = np.zeros(n_features+fit_intercept), np.zeros(n_samples)

    is_sparse = sparse.issparse(X)

    if len(w) != n_features + fit_intercept:
        raise ValueError(
            "The size of weights should be n_features + fit_intercept, \
                expected %i, got %i" % (n_features + fit_intercept, len(w)))

    for t in range(max_iter):
        if is_sparse:
            grad = datafit.full_grad_sparse(
                X.data, X.indptr, X.indices, y, Xw)
        else:
            grad = construct_grad(X, y, w[:n_features], Xw, datafit, all_feats)

        if ws_strategy == "subdiff":
            opt = penalty.subdiff_distance(w[:n_features], grad, all_feats)
        elif ws_strategy == "fixpoint":
            opt = dist_fix_point(w[:n_features], grad, datafit, penalty, all_feats)
        stop_crit = np.max(opt)
        if verbose:
            print(f"Stopping criterion max violation: {stop_crit:.2e}")
        if stop_crit <= tol:
            break
        # 1) select features : all unpenalized, + 2 * (nnz and penalized)
        ws_size = max(min(p0 + n_unpen, n_features),
                      min(2 * penalty.generalized_support(w[:n_features]).sum() -
                          n_unpen, n_features))

        opt[unpen] = np.inf  # always include unpenalized features
        opt[penalty.generalized_support(w[:n_features])] = np.inf

        # here use topk instead of np.argsort(opt)[-ws_size:]
        ws = np.argpartition(opt, -ws_size)[-ws_size:]

        # re init AA at every iter to consider ws
        accelerator = AndersonAcceleration(K=5)
        w_acc[:] = 0.

        if verbose:
            print(f'Iteration {t + 1}, {ws_size} feats in subpb.')

        # 2) do iterations on smaller problem
        is_sparse = sparse.issparse(X)
        for epoch in range(max_epochs):
            if is_sparse:
                _cd_epoch_sparse(
                    X.data, X.indptr, X.indices, y, w[:n_features], Xw,
                    datafit, penalty, ws)
            else:
                _cd_epoch(X, y, w[:n_features], Xw, datafit, penalty, ws)

            # update intercept
            if fit_intercept:
                intercept_old = w[-1]
                w[-1] -= datafit.intercept_update_step(y, Xw)
                Xw += (w[-1] - intercept_old)

            # 3) do Anderson acceleration on smaller problem
            # TODO optimize computation using ws
            w_acc[:], Xw_acc[:], is_extrapolated = accelerator.extrapolate(w, Xw)

            if is_extrapolated:  # avoid computing p_obj for un-extrapolated w, Xw
                # TODO : manage penalty.value(w, ws) for weighted Lasso
                p_obj = (datafit.value(y, w[:n_features], Xw) +
                         penalty.value(w[:n_features]))
                p_obj_acc = (datafit.value(y, w_acc[:n_features], Xw_acc) +
                             penalty.value(w_acc[:n_features]))

                if p_obj_acc < p_obj:
                    w[:], Xw[:] = w_acc, Xw_acc
                    p_obj = p_obj_acc

            if epoch % 10 == 0:
                if is_sparse:
                    grad_ws = construct_grad_sparse(
                        X.data, X.indptr, X.indices, y, w, Xw, datafit, ws)
                else:
                    grad_ws = construct_grad(X, y, w, Xw, datafit, ws)
                if ws_strategy == "subdiff":
                    opt_ws = penalty.subdiff_distance(w[:n_features], grad_ws, ws)
                elif ws_strategy == "fixpoint":
                    opt_ws = dist_fix_point(
                        w[:n_features], grad_ws, datafit, penalty, ws)

                stop_crit_in = np.max(opt_ws)
                if max(verbose - 1, 0):
                    p_obj = (datafit.value(y, w[:n_features], Xw) +
                             penalty.value(w[:n_features]))
                    print(f"Epoch {epoch + 1}, objective {p_obj:.10f}, "
                          f"stopping crit {stop_crit_in:.2e}")
                if ws_size == n_features:
                    if stop_crit_in <= tol:
                        break
                else:
                    if stop_crit_in < 0.3 * stop_crit:
                        if max(verbose - 1, 0):
                            print("Early exit")
                        break
        p_obj = datafit.value(y, w[:n_features], Xw) + penalty.value(w[:n_features])
        obj_out.append(p_obj)
    return w, np.array(obj_out), stop_crit


@njit
def _cd_epoch(X, y, w, Xw, datafit, penalty, ws):
    """Run an epoch of coordinate descent in place.

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        Design matrix.

    y : array, shape (n_samples,)
        Target vector.

    w : array, shape (n_features,)
        Coefficient vector.

    Xw : array, shape (n_samples,)
        Model fit.

    datafit : Datafit
        Datafit.

    penalty : Penalty
        Penalty.

    ws : array, shape (ws_size,)
        The range of features.
    """
    lc = datafit.lipschitz
    for j in ws:
        stepsize = 1/lc[j] if lc[j] != 0 else 1000
        Xj = X[:, j]
        old_w_j = w[j]
        w[j] = penalty.prox_1d(
            old_w_j - datafit.gradient_scalar(X, y, w, Xw, j) * stepsize,
            stepsize, j)
        if w[j] != old_w_j:
            Xw += (w[j] - old_w_j) * Xj


@njit
def _cd_epoch_sparse(X_data, X_indptr, X_indices, y, w, Xw, datafit, penalty, ws):
    """Run an epoch of coordinate descent in place for a sparse CSC array.

    Parameters
    ----------
    X_data : array, shape (n_elements,)
        `data` attribute of the sparse CSC matrix X.

    X_indptr : array, shape (n_features + 1,)
        `indptr` attribute of the sparse CSC matrix X.

    X_indices : array, shape (n_elements,)
        `indices` attribute of the sparse CSC matrix X.

    y : array, shape (n_samples,)
        Target vector.

    w : array, shape (n_features,)
        Coefficient vector.

    Xw : array, shape (n_samples,)
        Model fit.

    datafit : Datafit
        Datafit.

    penalty : Penalty
        Penalty.

    ws : array, shape (ws_size,)
        The working set.
    """
    lc = datafit.lipschitz
    for j in ws:
        stepsize = 1/lc[j] if lc[j] != 0 else 1000

        old_w_j = w[j]
        gradj = datafit.gradient_scalar_sparse(X_data, X_indptr, X_indices, y, Xw, j)
        w[j] = penalty.prox_1d(
            old_w_j - gradj * stepsize, stepsize, j)
        diff = w[j] - old_w_j
        if diff != 0:
            for i in range(X_indptr[j], X_indptr[j + 1]):
                Xw[X_indices[i]] += diff * X_data[i]
