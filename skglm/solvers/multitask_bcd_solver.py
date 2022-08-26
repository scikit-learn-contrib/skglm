import numpy as np

from scipy import sparse
from numba import njit
from numpy.linalg import norm
from sklearn.utils import check_array


def multitask_bcd_solver_path(
        X, Y, datafit, penalty, alphas=None,
        coef_init=None, max_iter=100, max_epochs=50_000, p0=10, tol=1e-6,
        use_acc=True, return_n_iter=False, ws_strategy="subdiff", verbose=0):
    r"""Compute optimization path for multi-task optimization problem.

    The loss is customized by passing various choices of datafit and penalty:
    loss = datafit.value() + penalty.value()

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        Training data.

    Y : array, shape (n_samples, n_tasks)
        Target matrix.

    datafit : instance of BaseMultiTaskDatafit
        Datafitting term.

    penalty : instance of BasePenalty
        Penalty used in the model.

    alphas : ndarray, optional
        List of alphas where to compute the models.
        If ``None`` alphas are set automatically.

    coef_init : ndarray, shape (n_features, n_tasks) | None, optional, (default=None)
        Initial value of coefficients. If None, np.zeros(n_features, n_tasks) is used.

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

    ws_strategy : str, optional
        The score used to build the working set. Can be 'subdiff' or 'fixpoint'.

    verbose : bool or int, optional
        Amount of verbosity. 0/False is silent.

    Returns
    -------
    alphas : array, shape (n_alphas,)
        The alphas along the path where models are computed.

    coefs : array, shape (n_features, n_tasks, n_alphas)
        Coefficients along the path.

    stop_crit : array, shape (n_alphas,)
        Value of stopping criterion at convergence along the path.

    n_iters : array, shape (n_alphas,), optional
        The number of iterations along the path.
    """
    X = check_array(X, "csc", dtype=[
        np.float64, np.float32], order="F", copy=False)
    Y = check_array(Y, "csc", dtype=[
        np.float64, np.float32], order="F", copy=False)
    if sparse.issparse(X):
        datafit.initialize_sparse(X.data, X.indptr, X.indices, Y)
    else:
        datafit.initialize(X, Y)
    n_features = X.shape[1]
    n_tasks = Y.shape[1]
    if alphas is None:
        raise ValueError("alphas should be provided.")
        # alpha_max = np.max(norm(X.T @ Y, ord=2, axis=1)) / n_samples
        # alphas = alpha_max * \
        # np.geomspace(1, eps, n_alphas, dtype=X.dtype)
    # else:
        # alphas = np.sort(alphas)[::-1]

    n_alphas = len(alphas)

    coefs = np.zeros((n_features, n_tasks, n_alphas), order="C",
                     dtype=X.dtype)
    stop_crits = np.zeros(n_alphas)

    if return_n_iter:
        n_iters = np.zeros(n_alphas, dtype=int)

    Y = np.asfortranarray(Y)
    XW = np.zeros(Y.shape, order='F')
    for t in range(n_alphas):
        alpha = alphas[t]
        penalty.alpha = alpha  # TODO this feels it will break sklearn compat
        if verbose:
            msg = "##### Computing alpha %d/%d" % (t + 1, n_alphas)
            print("#" * len(msg))
            print(msg)
            print("#" * len(msg))
        if t > 0:
            W = coefs[:, :, t - 1].copy()
            p_t = max(len(np.where(W[:, 0] != 0)[0]), p0)
        else:
            if coef_init is not None:
                W = coef_init.T
                XW = np.asfortranarray(X @ W)
                p_t = max(len(np.where(W[:, 0] != 0)[0]), p0)
            else:
                W = np.zeros(
                    (n_features, n_tasks), dtype=X.dtype, order='C')
                p_t = 10
        sol = multitask_bcd_solver(
            X, Y, datafit, penalty, W, XW, p0=p_t,
            tol=tol, max_iter=max_iter, max_epochs=max_epochs,
            verbose=verbose, use_acc=use_acc, ws_strategy=ws_strategy)
        coefs[:, :, t], stop_crits[t] = sol[0], sol[2]

        if return_n_iter:
            n_iters[t] = len(sol[1])

    coefs = np.swapaxes(coefs, 0, 1).copy('F')

    results = alphas, coefs, stop_crits
    if return_n_iter:
        results += (n_iters,)

    return results


def multitask_bcd_solver(
        X, Y, datafit, penalty, W, XW, max_iter=50, max_epochs=50_000, p0=10,
        tol=1e-4, use_acc=True, K=5, ws_strategy="subdiff", verbose=0):
    r"""Run a multitask block coordinate descent solver.

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        Training data.

    Y : array, shape (n_samples, n_tasks)
        Target matrix.

    datafit : instance of BaseMultiTaskDatafit
        Datafitting term.

    penalty : instance of BasePenalty
        Penalty used in the model.

    W : array, shape (n_features, n_tasks)
        Coefficient matrix.

    XW : ndarray, shape (n_samples, n_tasks)
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

    ws_strategy : str, ('subdiff'|'fixpoint'), optional
        The score used to build the working set.

    verbose : bool or int, optional
        Amount of verbosity. 0/False is silent.

    Returns
    -------
    coefs : array, shape (n_features, n_tasks, n_alphas)
        Coefficients along the path.

    obj_out : array, shape (n_iter,)
        The objective values at every outer iteration.

    stop_crit : float
        Value of stopping criterion at convergence.
    """
    n_tasks = Y.shape[1]
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
                X.data, X.indptr, X.indices, Y, XW)
        else:
            grad = construct_grad(X, Y, W, XW, datafit, all_feats)

        if ws_strategy == "subdiff":
            opt = penalty.subdiff_distance(W, grad, all_feats)
        elif ws_strategy == "fixpoint":
            opt = dist_fix_point(W, grad, datafit, penalty, all_feats)
        stop_crit = np.max(opt)
        if verbose:
            print(f"Stopping criterion max violation: {stop_crit:.2e}")
        if stop_crit <= tol:
            break
        # 1) select features : all unpenalized, + 2 * (nnz and penalized)
        ws_size = max(p0 + n_unpen,
                      min(2 * (norm(W, axis=1) != 0).sum() - n_unpen,
                          n_features))
        opt[unpen] = np.inf  # always include unpenalized features
        opt[norm(W, axis=1) != 0] = np.inf  # TODO check
        ws = np.argpartition(opt, -ws_size)[-ws_size:]
        # is equivalent to ws = np.argsort(kkt)[-ws_size:]

        if use_acc:
            last_K_w = np.zeros([K + 1, ws_size * n_tasks])
            U = np.zeros([K, ws_size * n_tasks])

        if verbose:
            print(f'Iteration {t + 1}, {ws_size} feats in subpb.')

        # 2) do iterations on smaller problem
        is_sparse = sparse.issparse(X)
        for epoch in range(max_epochs):
            if is_sparse:
                _bcd_epoch_sparse(
                    X.data, X.indptr, X.indices, Y, W, XW, datafit, penalty,
                    ws)
            else:
                _bcd_epoch(X, Y, W, XW, datafit, penalty, ws)
            if use_acc:
                last_K_w[epoch % (K + 1)] = W[ws, :].ravel()

                # 3) do Anderson acceleration on smaller problem
                if epoch % (K + 1) == K:
                    for k in range(K):
                        U[k] = last_K_w[k + 1] - last_K_w[k]
                    C = np.dot(U, U.T)

                    try:
                        z = np.linalg.solve(C, np.ones(K))
                        c = z / z.sum()
                        W_acc = np.zeros((n_features, n_tasks))
                        W_acc[ws, :] = np.sum(
                            last_K_w[:-1] * c[:, None], axis=0).reshape(
                                (ws_size, n_tasks))
                        p_obj = datafit.value(Y, W, XW) + penalty.value(W)
                        Xw_acc = X[:, ws] @ W_acc[ws]
                        p_obj_acc = datafit.value(
                            Y, W_acc, Xw_acc) + penalty.value(W_acc)
                        if p_obj_acc < p_obj:
                            W[:] = W_acc
                            XW[:] = Xw_acc
                    except np.linalg.LinAlgError:
                        if max(verbose - 1, 0):
                            print("----------Linalg error")

            if epoch > 0 and epoch % 10 == 0:
                p_obj = datafit.value(Y, W[ws, :], XW) + penalty.value(W)

                if is_sparse:
                    grad_ws = construct_grad_sparse(
                        X.data, X.indptr, X.indices, Y, XW, datafit, ws)
                else:
                    grad_ws = construct_grad(X, Y, W, XW, datafit, ws)

                if ws_strategy == "subdiff":
                    opt_ws = penalty.subdiff_distance(W, grad_ws, ws)
                elif ws_strategy == "fixpoint":
                    opt_ws = dist_fix_point(W, grad_ws, datafit, penalty, ws)

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
    return W, np.array(obj_out), stop_crit


@njit
def dist_fix_point(W, grad_ws, datafit, penalty, ws):
    """Compute the violation of the fixed point iterate schema.

    Parameters
    ----------
    W : array, shape (n_features, n_tasks)
        Coefficient matrix.

    grad_ws : array, shape (ws_size, n_tasks)
        Gradient restricted to the working set.

    datafit: instance of BaseMultiTaskDatafit
        Datafit.

    penalty: instance of BasePenalty
        Penalty.

    ws : array, shape (ws_size,)
        The working set.

    Returns
    -------
    dist_fix_point : array, shape (ws_size,)
        Contain the violation score for every feature.
    """
    dist_fix_point = np.zeros(ws.shape[0])
    for idx, j in enumerate(ws):
        lcj = datafit.lipschitz[j]
        if lcj:
            dist_fix_point[idx] = norm(
                W[j] - penalty.prox_1feat(W[j] - grad_ws[idx] / lcj, 1. / lcj, j))
    return dist_fix_point


@njit
def construct_grad(X, Y, W, XW, datafit, ws):
    """Compute the gradient of the datafit restricted to the working set.

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        Design matrix.

    Y : array, shape (n_samples, n_tasks)
        Target matrix.

    W : array, shape (n_features, n_tasks)
        Coefficient matrix.

    XW : array, shape (n_samples, n_tasks)
        Model fit.

    datafit : instance of BaseMultiTaskDatafit
        Datafit.

    ws : array, shape (ws_size,)
        The working set.

    Returns
    -------
    grad : array, shape (ws_size, n_tasks)
        The gradient restricted to the working set.
    """
    n_tasks = XW.shape[1]
    grad = np.zeros((ws.shape[0], n_tasks))
    for idx, j in enumerate(ws):
        grad[idx, :] = datafit.gradient_j(X, Y, W, XW, j)
    return grad


@njit
def construct_grad_sparse(data, indptr, indices, Y, XW, datafit, ws):
    """Compute the gradient of the datafit restricted to the working set.

    Parameters
    ----------
    data : array-like
        Data array of the matrix in CSC format.

    indptr : array-like
        CSC format index point array.

    indices : array-like
        CSC format index array.

    Y : array, shape (n_samples, n_tasks)
        Target matrix.

    XW : array, shape (n_samples, n_tasks)
        Model fit.

    datafit : instance of BaseMultiTaskDatafit
        Datafit.

    ws : array, shape (ws_size,)
        The working set.

    Returns
    -------
    grad : array, shape (ws_size, n_tasks)
        The gradient restricted to the working set.
    """
    n_tasks = XW.shape[1]
    grad = np.zeros((ws.shape[0], n_tasks))
    for idx, j in enumerate(ws):
        grad[idx, :] = datafit.gradient_j_sparse(
            data, indptr, indices, Y, XW, j)
    return grad


@njit
def _bcd_epoch(X, Y, W, XW, datafit, penalty, ws):
    """Run an epoch of block coordinate descent in place.

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        Design matrix.

    Y : array, shape (n_samples, n_tasks)
        Target matrix.

    W : array, shape (n_features, n_tasks)
        Coefficient matrix.

    XW : array, shape (n_samples, n_tasks)
        Model fit.

    datafit : instance of BaseMultiTaskDatafit
        Datafit.

    penalty : instance of BasePenalty
        Penalty.

    ws : array, shape (ws_size,)
        The working set.
    """
    lc = datafit.lipschitz
    n_tasks = Y.shape[1]
    for j in ws:
        if lc[j] == 0.:
            continue
        Xj = X[:, j]
        old_W_j = W[j, :].copy()  # copy is very important here
        W[j:j+1, :] = penalty.prox_1feat(
            W[j:j+1, :] - datafit.gradient_j(X, Y, W, XW, j) / lc[j],
            1 / lc[j], j)
        if not np.all(W[j, :] == old_W_j):
            for k in range(n_tasks):
                tmp = W[j, k] - old_W_j[k]
                if tmp != 0:
                    XW[:, k] += tmp * Xj


@njit
def _bcd_epoch_sparse(X_data, X_indptr, X_indices, Y, W, XW, datafit, penalty, ws):
    """Run an epoch of block coordinate descent in place for a sparse CSC array.

    Parameters
    ----------
    X_data : array, shape (n_elements,)
        `data` attribute of the sparse CSC matrix X.

    X_indptr : array, shape (n_features + 1,)
        `indptr` attribute of the sparse CSC matrix X.

    X_indices : array, shape (n_elements,)
        `indices` attribute of the sparse CSC matrix X.

    Y : array, shape (n_samples, n_tasks)
        Target matrix.

    W : array, shape (n_features, n_tasks)
        Coefficient matrix.

    XW : array, shape (n_samples, n_tasks)
        Model fit.

    datafit : instance of BaseMultiTaskDatafit
        Datafit.

    penalty : instance of BasePenalty
        Penalty.

    ws : array, shape (ws_size,)
        Features to be updated.
    """
    lc = datafit.lipschitz
    for j in ws:
        if lc[j] == 0.:
            continue
        old_W_j = W[j, :].copy()
        grad_j = datafit.gradient_j_sparse(X_data, X_indptr, X_indices, Y, XW, j)
        W[j] = penalty.prox_1feat(
            old_W_j - grad_j / lc[j], 1 / lc[j], j)
        # TODO: could be enhanced?
        diff = W[j, :] - old_W_j
        if not np.all(diff == 0):
            for i in range(X_indptr[j], X_indptr[j + 1]):
                for t in range(Y.shape[1]):
                    XW[X_indices[i], t] += diff[t] * X_data[i]
