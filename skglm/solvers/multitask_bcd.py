import numpy as np
from scipy import sparse
from numba import njit
from numpy.linalg import norm
from sklearn.utils import check_array
from skglm.solvers.base import BaseSolver


class MultiTaskBCD(BaseSolver):
    """Block coordinate descent solver for multi-task problems."""

    def __init__(self, max_iter=100, max_epochs=50_000, p0=10, tol=1e-6,
                 use_acc=True, ws_strategy="subdiff", fit_intercept=True,
                 warm_start=False, verbose=0):
        self.max_iter = max_iter
        self.max_epochs = max_epochs
        self.p0 = p0
        self.tol = tol
        self.use_acc = use_acc
        self.ws_strategy = ws_strategy
        self.fit_intercept = fit_intercept
        self.warm_start = warm_start
        self.verbose = verbose

    def solve(self, X, Y, datafit, penalty, W_init=None, XW_init=None):
        n_samples, n_features = X.shape
        n_tasks = Y.shape[1]
        pen = penalty.is_penalized(n_features)
        unpen = ~pen
        n_unpen = unpen.sum()
        obj_out = []
        all_feats = np.arange(n_features)
        stop_crit = np.inf  # initialize for case n_iter=0
        K = 5

        W = (np.zeros((n_features + self.fit_intercept, n_tasks)) if W_init is None
             else W_init)
        XW = np.zeros((n_samples, n_tasks)) if XW_init is None else XW_init

        if W.shape[0] != n_features + self.fit_intercept:
            if self.fit_intercept:
                val_error_message = (
                    "W.shape[0] should be n_features + 1 when using fit_intercept=True:"
                    f" expected {n_features + 1}, got {W.shape[0]}.")
            else:
                val_error_message = (
                    "W.shape[0] should be of size n_features: "
                    f"expected {n_features}, got {W.shape[0]}.")
            raise ValueError(val_error_message)

        is_sparse = sparse.issparse(X)
        if is_sparse:
            datafit.initialize_sparse(X.data, X.indptr, X.indices, Y)
            lipschitz = datafit.get_lipschitz_sparse(X.data, X.indptr, X.indices, Y)
        else:
            datafit.initialize(X, Y)
            lipschitz = datafit.get_lipschitz(X, Y)

        for t in range(self.max_iter):
            if is_sparse:
                grad = datafit.full_grad_sparse(
                    X.data, X.indptr, X.indices, Y, XW)
            else:
                grad = construct_grad(X, Y, W, XW, datafit, all_feats)

            if self.ws_strategy == "subdiff":
                opt = penalty.subdiff_distance(W, grad, all_feats)
            elif self.ws_strategy == "fixpoint":
                opt = dist_fix_point_bcd(W, grad, datafit, penalty, all_feats)
            stop_crit = np.max(opt)
            if self.verbose:
                print(f"Stopping criterion max violation: {stop_crit:.2e}")
            if stop_crit <= self.tol:
                break
            # 1) select features : all unpenalized, + 2 * (nnz and penalized)
            # TODO fix p0 takes the intercept into account
            ws_size = min(n_features, max(2 * (norm(W, axis=1) != 0).sum() - n_unpen,
                                          self.p0 + n_unpen))
            opt[unpen] = np.inf  # always include unpenalized features
            opt[norm(W[:n_features], axis=1) != 0] = np.inf  # TODO check
            ws = np.argpartition(opt, -ws_size)[-ws_size:]
            # is equivalent to ws = np.argsort(kkt)[-ws_size:]

            if self.use_acc:
                last_K_w = np.zeros([K + 1,
                                     (ws_size + self.fit_intercept) * n_tasks])
                U = np.zeros([K, (ws_size + self.fit_intercept) * n_tasks])

            if self.verbose:
                print(f'Iteration {t + 1}, {ws_size} feats in subpb.')

            # 2) do iterations on smaller problem
            is_sparse = sparse.issparse(X)
            for epoch in range(self.max_epochs):
                if is_sparse:
                    _bcd_epoch_sparse(
                        X.data, X.indptr, X.indices, Y, W, XW,
                        lipschitz, datafit, penalty, ws
                    )
                else:
                    _bcd_epoch(X, Y, W, XW, lipschitz, datafit, penalty, ws)

                # update intercept
                if self.fit_intercept:
                    intercept_old = W[-1, :].copy()
                    W[-1, :] -= datafit.intercept_update_step(Y, XW)
                    XW += (W[-1, :] - intercept_old)

                if self.use_acc:
                    if self.fit_intercept:
                        ws_ = np.append(ws, -1)
                    else:
                        ws_ = ws.copy()
                    last_K_w[epoch % (K + 1)] = W[ws_, :].ravel()

                    # 3) do Anderson acceleration on smaller problem
                    if epoch % (K + 1) == K:
                        for k in range(K):
                            U[k] = last_K_w[k + 1] - last_K_w[k]
                        C = np.dot(U, U.T)

                        try:
                            z = np.linalg.solve(C, np.ones(K))
                            c = z / z.sum()
                            W_acc = np.zeros((n_features + self.fit_intercept, n_tasks))
                            W_acc[ws_, :] = np.sum(
                                last_K_w[:-1] * c[:, None], axis=0).reshape(
                                    (ws_size + self.fit_intercept, n_tasks))
                            p_obj = datafit.value(Y, W, XW) + penalty.value(W)
                            Xw_acc = (X[:, ws] @ W_acc[ws]
                                      + self.fit_intercept * W_acc[-1])
                            p_obj_acc = datafit.value(
                                Y, W_acc, Xw_acc) + penalty.value(W_acc)
                            if p_obj_acc < p_obj:
                                W[:] = W_acc
                                XW[:] = Xw_acc
                        except np.linalg.LinAlgError:
                            if max(self.verbose - 1, 0):
                                print("----------Linalg error")

                if epoch > 0 and epoch % 10 == 0:
                    p_obj = datafit.value(Y, W[ws, :], XW) + penalty.value(W)

                    if is_sparse:
                        grad_ws = construct_grad_sparse(
                            X.data, X.indptr, X.indices, Y, XW, datafit, ws)
                    else:
                        grad_ws = construct_grad(X, Y, W, XW, datafit, ws)

                    if self.ws_strategy == "subdiff":
                        opt_ws = penalty.subdiff_distance(W, grad_ws, ws)
                    elif self.ws_strategy == "fixpoint":
                        opt_ws = dist_fix_point_bcd(
                            W, grad_ws, lipschitz[ws], datafit, penalty, ws
                        )

                    stop_crit_in = np.max(opt_ws)
                    if max(self.verbose - 1, 0):
                        print(f"Epoch {epoch + 1}, objective {p_obj:.10f}, "
                              f"stopping crit {stop_crit_in:.2e}")
                    if ws_size == n_features:
                        if stop_crit_in <= self.tol:
                            break
                    else:
                        if stop_crit_in < 0.3 * stop_crit:
                            if max(self.verbose - 1, 0):
                                print("Early exit")
                            break
            obj_out.append(p_obj)
        return W, np.array(obj_out), stop_crit

    def path(self, X, Y, datafit, penalty, alphas, W_init=None, return_n_iter=False):
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
        n_alphas = len(alphas)

        coefs = np.zeros((n_features + self.fit_intercept, n_tasks, n_alphas),
                         order="C", dtype=X.dtype)
        stop_crits = np.zeros(n_alphas)
        p0 = self.p0

        if return_n_iter:
            n_iters = np.zeros(n_alphas, dtype=int)

        Y = np.asfortranarray(Y)
        XW = np.zeros(Y.shape, order='F')
        for t in range(n_alphas):
            alpha = alphas[t]
            penalty.alpha = alpha  # TODO this feels it will break sklearn compat
            if self.verbose:
                msg = "##### Computing alpha %d/%d" % (t + 1, n_alphas)
                print("#" * len(msg))
                print(msg)
                print("#" * len(msg))
            if t > 0:
                W = coefs[:, :, t - 1].copy()
                p0 = max(len(np.where(W[:, 0] != 0)[0]), p0)
            else:
                if W_init is not None:
                    W = W_init.T
                    XW = np.asfortranarray(X @ W)
                    p0 = max(len(np.where(W[:, 0] != 0)[0]), p0)
                else:
                    W = np.zeros(
                        (n_features + self.fit_intercept, n_tasks), dtype=X.dtype,
                        order='C')
                    p0 = 10
            sol = self.solve(X, Y, datafit, penalty, W, XW)
            coefs[:, :, t], stop_crits[t] = sol[0], sol[2]

            if return_n_iter:
                n_iters[t] = len(sol[1])

        coefs = np.swapaxes(coefs, 0, 1).copy('F')

        results = alphas, coefs, stop_crits
        if return_n_iter:
            results += (n_iters,)

        return results


@njit
def dist_fix_point_bcd(W, grad_ws, lipschitz_ws, datafit, penalty, ws):
    """Compute the violation of the fixed point iterate schema.

    Parameters
    ----------
    W : array, shape (n_features, n_tasks)
        Coefficient matrix.

    grad_ws : array, shape (len(ws), n_tasks)
        Gradient restricted to the working set.

    datafit: instance of BaseMultiTaskDatafit
        Datafit.

    lipschitz_ws :  array, shape (len(ws),)
        Blockwise gradient Lipschitz constants, restricted to working set.

    penalty: instance of BasePenalty
        Penalty.

    ws : array, shape (len(ws),)
        The working set.

    Returns
    -------
    dist : array, shape (ws_size,)
        Contain the violation score for every feature.
    """
    dist = np.zeros(ws.shape[0])

    for idx, j in enumerate(ws):
        if lipschitz_ws[idx] == 0.:
            continue

        step_j = 1 / lipschitz_ws[idx]
        dist[idx] = norm(
            W[j] - penalty.prox_1feat(W[j] - step_j * grad_ws[idx], step_j, j)
        )
    return dist


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
def _bcd_epoch(X, Y, W, XW, lc, datafit, penalty, ws):
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

    lc :  array, shape (n_features,)
        Blockwise gradient Lipschitz constants.

    datafit : instance of BaseMultiTaskDatafit
        Datafit.

    penalty : instance of BasePenalty
        Penalty.

    ws : array, shape (ws_size,)
        The working set.
    """
    n_tasks = Y.shape[1]
    for j in ws:
        if lc[j] == 0.:
            continue
        Xj = X[:, j]
        old_W_j = W[j, :].copy()  # copy is very important here
        W[j, :] = penalty.prox_1feat(
            W[j, :] - datafit.gradient_j(X, Y, W, XW, j) / lc[j],
            1 / lc[j], j)
        if not np.all(W[j, :] == old_W_j):
            for k in range(n_tasks):
                tmp = W[j, k] - old_W_j[k]
                if tmp != 0:
                    XW[:, k] += tmp * Xj


@njit
def _bcd_epoch_sparse(X_data, X_indptr, X_indices, Y, W, XW, lc, datafit, penalty, ws):
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

    lc :  array, shape (n_features,)
        Blockwise gradient Lipschitz constants.

    datafit : instance of BaseMultiTaskDatafit
        Datafit.

    penalty : instance of BasePenalty
        Penalty.

    ws : array, shape (ws_size,)
        Features to be updated.
    """
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
