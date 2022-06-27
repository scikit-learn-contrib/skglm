import numpy as np
from numba import njit
from scipy import sparse
from sklearn.utils import check_array
from skglm.solvers.common import construct_grad, construct_grad_sparse, dist_fix_point


def cd_solver_path(X, y, datafit, penalty, alphas=None,
                   coef_init=None, max_iter=20, max_epochs=50_000,
                   p0=10, tol=1e-4, use_acc=True, return_n_iter=False,
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

    coef_init : ndarray, shape (n_features,) | None, optional, (default=None)
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

    use_acc : bool, optional
        Usage of Anderson acceleration for faster convergence.

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

    coefs : array, shape (n_features, n_alphas)
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

    # if X_offset is not None:
    #     X_sparse_scaling = X_offset / X_scale
    #     X_sparse_scaling = np.asarray(X_sparse_scaling, dtype=X.dtype)
    # else:
    #     X_sparse_scaling = np.zeros(n_features, dtype=X.dtype)

    # X_dense, X_data, X_indices, X_indptr = _sparse_and_dense(X)

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

    coefs = np.zeros((n_features, n_alphas), order='F', dtype=X.dtype)
    stop_crits = np.zeros(n_alphas)

    if return_n_iter:
        n_iters = np.zeros(n_alphas, dtype=int)

    for t in range(n_alphas):

        alpha = alphas[t]
        penalty.alpha = alpha  # TODO this feels it will break sklearn compat
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
                supp_size = penalty.generalized_support(w).sum()
                p0 = max(supp_size, p0)
                if supp_size:
                    Xw = X @ w
                # TODO explain/clean this hack
                else:
                    Xw = np.zeros_like(y)
            else:
                w = np.zeros(n_features, dtype=X.dtype)
                Xw = np.zeros(X.shape[0], dtype=X.dtype)

        sol = cd_solver(
            X, y, datafit, penalty, w, Xw,
            max_iter=max_iter, max_epochs=max_epochs, p0=p0, tol=tol,
            use_acc=use_acc, verbose=verbose, ws_strategy=ws_strategy)

        coefs[:, t] = w
        stop_crits[t] = sol[-1]

        if return_n_iter:
            n_iters[t] = len(sol[1])

    results = alphas, coefs, stop_crits
    if return_n_iter:
        results += (n_iters,)

    return results


def cd_solver(
        X, y, datafit, penalty, w, Xw, max_iter=50, max_epochs=50_000, p0=10,
        tol=1e-4, use_acc=True, K=5, ws_strategy="subdiff", verbose=0):
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

    max_iter : int, optional
        The maximum number of iterations (definition of working set and
        resolution of problem restricted to features in working set).

    max_epochs : int, optional
        Maximum number of (block) CD epochs on each subproblem.

    p0 : int, optional
        First working set size.

    tol : float, optional
        The tolerance for the optimization.

    use_acc : bool, optional
        Usage of Anderson acceleration for faster convergence.

    K : int, optional
        The number of past primal iterates used to build an extrapolated point.

    ws_strategy : ('subdiff'|'fixpoint'), optional
        The score used to build the working set.

    verbose : bool or int, optional
        Amount of verbosity. 0/False is silent.

    Returns
    -------
    coefs : array, shape (n_features, n_alphas)
        Coefficients along the path.

    obj_out : array, shape (n_iter,)
        The objective values at every outer iteration.

    stop_crit : float
        Value of stopping criterion at convergence.
    """
    if ws_strategy not in ("subdiff", "fixpoint"):
        raise ValueError(f'Unsupported value for ws_strategy: {ws_strategy}')
    n_features = X.shape[1]
    pen = penalty.is_penalized(n_features)
    unpen = ~pen
    n_unpen = unpen.sum()
    obj_out = []
    all_feats = np.arange(n_features)
    stop_crit = np.inf  # initialize for case n_iter=0

    is_sparse = sparse.issparse(X)
    for t in range(max_iter):
        if is_sparse:
            grad = datafit.full_grad_sparse(
                X.data, X.indptr, X.indices, y, Xw)
        else:
            grad = construct_grad(X, y, w, Xw, datafit, all_feats)

        if ws_strategy == "subdiff":
            opt = penalty.subdiff_distance(w, grad, all_feats)
        elif ws_strategy == "fixpoint":
            opt = dist_fix_point(w, grad, datafit, penalty, all_feats)
        stop_crit = np.max(opt)
        if verbose:
            print(f"Stopping criterion max violation: {stop_crit:.2e}")
        if stop_crit <= tol:
            break
        # 1) select features : all unpenalized, + 2 * (nnz and penalized)
        ws_size = max(min(p0 + n_unpen, n_features),
                      min(2 * penalty.generalized_support(w).sum() -
                          n_unpen, n_features))

        opt[unpen] = np.inf  # always include unpenalized features
        opt[penalty.generalized_support(w)] = np.inf

        # here use topk instead of sorting the full array
        # ie the following line
        ws = np.argpartition(opt, -ws_size)[-ws_size:]
        # is equivalent to ws = np.argsort(opt)[-ws_size:]

        if use_acc:
            last_K_w = np.zeros([K + 1, ws_size])
            U = np.zeros([K, ws_size])

        if verbose:
            print(f'Iteration {t + 1}, {ws_size} feats in subpb.')

        # 2) do iterations on smaller problem
        is_sparse = sparse.issparse(X)
        for epoch in range(max_epochs):
            if is_sparse:
                _cd_epoch_sparse(
                    X.data, X.indptr, X.indices, y, w, Xw, datafit, penalty,
                    ws)
            else:
                _cd_epoch(X, y, w, Xw, datafit, penalty, ws)

            # 3) do Anderson acceleration on smaller problem
            # TODO optimize computation using ws
            if use_acc:
                last_K_w[epoch % (K + 1)] = w[ws]

                if epoch % (K + 1) == K:
                    for k in range(K):
                        U[k] = last_K_w[k + 1] - last_K_w[k]
                    C = np.dot(U, U.T)

                    try:
                        z = np.linalg.solve(C, np.ones(K))
                        # When C is ill-conditioned, z can take very large finite
                        # positive and negative values (1e35 and -1e35), which leads
                        # to z.sum() being null.
                        if z.sum() == 0:
                            raise np.linalg.LinAlgError
                    except np.linalg.LinAlgError:
                        if max(verbose - 1, 0):
                            print("----------Linalg error")
                    else:
                        c = z / z.sum()
                        w_acc = np.zeros(n_features)
                        w_acc[ws] = np.sum(
                            last_K_w[:-1] * c[:, None], axis=0)
                        # TODO create a p_obj function ?
                        # TODO : managed penalty.value(w[ws])
                        p_obj = datafit.value(y, w, Xw) + penalty.value(w)
                        # p_obj = datafit.value(y, w, Xw) +penalty.value(w[ws])
                        Xw_acc = X[:, ws] @ w_acc[ws]
                        # TODO : managed penalty.value(w[ws])
                        p_obj_acc = datafit.value(
                            y, w_acc, Xw_acc) + penalty.value(w_acc)
                        if p_obj_acc < p_obj:
                            w[:] = w_acc
                            Xw[:] = Xw_acc

            if epoch % 10 == 0:
                # TODO : manage penalty.value(w, ws) for weighted Lasso
                p_obj = datafit.value(y, w[ws], Xw) + penalty.value(w)

                if is_sparse:
                    grad = construct_grad_sparse(
                        X.data, X.indptr, X.indices, y, w, Xw, datafit, ws)
                else:
                    grad = construct_grad(X, y, w, Xw, datafit, ws)
                if ws_strategy == "subdiff":
                    opt_ws = penalty.subdiff_distance(w, grad, ws)
                elif ws_strategy == "fixpoint":
                    opt_ws = dist_fix_point(w, grad, datafit, penalty, ws)

                stop_crit_in = np.max(opt_ws)
                if max(verbose - 1, 0):
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
        obj_out.append(p_obj)
    return w, np.array(obj_out), stop_crit


@njit
def _cd_epoch(X, y, w, Xw, datafit, penalty, feats):
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

    feats : array, shape (n_features,)
        The range of features.
    """
    lc = datafit.lipschitz
    for j in feats:
        stepsize = 1/lc[j] if lc[j] != 0 else 1000
        Xj = X[:, j]
        old_w_j = w[j]
        w[j] = penalty.prox_1d(
            old_w_j - datafit.gradient_scalar(X, y, w, Xw, j) * stepsize,
            stepsize, j)
        if w[j] != old_w_j:
            Xw += (w[j] - old_w_j) * Xj


@njit
def _cd_epoch_sparse(X_data, X_indptr, X_indices, y, w, Xw, datafit, penalty, feats):
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

    feats : array, shape (n_features,)
        The range of features.
    """
    lc = datafit.lipschitz
    for j in feats:
        stepsize = 1/lc[j] if lc[j] != 0 else 1000

        old_w_j = w[j]
        gradj = datafit.gradient_scalar_sparse(X_data, X_indptr, X_indices, y, Xw, j)
        w[j] = penalty.prox_1d(
            old_w_j - gradj * stepsize, stepsize, j)
        diff = w[j] - old_w_j
        if diff != 0:
            for i in range(X_indptr[j], X_indptr[j + 1]):
                Xw[X_indices[i]] += diff * X_data[i]
