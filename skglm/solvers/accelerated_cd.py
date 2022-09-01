import numpy as np
from numba import njit
from scipy import sparse
from sklearn.utils import check_array
from skglm.solvers.common import construct_grad, construct_grad_sparse, dist_fix_point
from skglm.utils import AndersonAcceleration


class AcceleratedCD:
    """Coordinate descent solver with working sets and Anderson acceleration.

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

    References
    ----------
    .. [1] Bertrand, Q. and Klopfenstein, Q. and Bannier, P.-A. and Gidel, G.
           and Massias, M.
           "Beyond L1: Faster and Better Sparse Models with skglm", 2022
           https://arxiv.org/abs/2204.07826

    .. [2] Bertrand, Q. and Massias, M.
           "Anderson acceleration of coordinate descent", AISTATS, 2021
           https://proceedings.mlr.press/v130/bertrand21a.html
           code: https://github.com/mathurinm/andersoncd
    """

    def __init__(self, max_iter=50, max_epochs=50_000, p0=10,
                 tol=1e-4, ws_strategy="subdiff", verbose=0):
        self.max_iter = max_iter
        self.max_epochs = max_epochs
        self.p0 = p0
        self.tol = tol
        self.ws_strategy = ws_strategy
        self.verbose = verbose

    def solve(self, X, y, datafit, penalty, w_init=None, Xw_init=None):
        if self.ws_strategy not in ("subdiff", "fixpoint"):
            raise ValueError(
                'Unsupported value for self.ws_strategy:', self.ws_strategy)

        n_samples, n_features = X.shape
        w = np.zeros(n_features) if w_init is None else w_init
        Xw = np.zeros(n_samples) if Xw_init is None else Xw_init
        pen = penalty.is_penalized(n_features)
        unpen = ~pen
        n_unpen = unpen.sum()
        obj_out = []
        all_feats = np.arange(n_features)
        stop_crit = np.inf  # initialize for case n_iter=0
        w_acc, Xw_acc = np.zeros(n_features + self.fit_intercept), np.zeros(n_samples)

        is_sparse = sparse.issparse(X)
        if is_sparse:
            datafit.initialize_sparse(X.data, X.indptr, X.indices, y)
        else:
            datafit.initialize(X, y)

        if len(w) != n_features + self.fit_intercept:
            if self.fit_intercept:
                val_error_message = (
                    "Inconsistent size of coefficients with n_features + 1\n"
                    f"expected {n_features + 1}, got {len(w)}")
            else:
                val_error_message = (
                    "Inconsistent size of coefficients with n_features\n"
                    f"expected {n_features}, got {len(w)}")
            raise ValueError(val_error_message)

        for t in range(self.max_iter):
            if is_sparse:
                grad = datafit.full_grad_sparse(
                    X.data, X.indptr, X.indices, y, Xw)
            else:
                grad = construct_grad(X, y, w[:n_features], Xw, datafit, all_feats)

            # The intercept is not taken into account in the optimality conditions since
            # the derivative w.r.t. to the intercept may be very large. It is not likely
            # to change significantly the optimality conditions.
            if self.ws_strategy == "subdiff":
                opt = penalty.subdiff_distance(w[:n_features], grad, all_feats)
            elif self.ws_strategy == "fixpoint":
                opt = dist_fix_point(w[:n_features], grad, datafit, penalty, all_feats)

            if self.fit_intercept:
                intercept_opt = np.abs(datafit.intercept_update_step(y, Xw))
            else:
                intercept_opt = 0.

            stop_crit = max(np.max(opt), intercept_opt)

            if self.verbose:
                print(f"Stopping criterion max violation: {stop_crit:.2e}")
            if stop_crit <= self.tol:
                break
            # 1) select features : all unpenalized, + 2 * (nnz and penalized)
            ws_size = max(min(self.p0 + n_unpen, n_features),
                          min(2 * penalty.generalized_support(w[:n_features]).sum() -
                              n_unpen, n_features))

            opt[unpen] = np.inf  # always include unpenalized features
            opt[penalty.generalized_support(w[:n_features])] = np.inf

            # here use topk instead of np.argsort(opt)[-ws_size:]
            ws = np.argpartition(opt, -ws_size)[-ws_size:]

            # re init AA at every iter to consider ws
            accelerator = AndersonAcceleration(K=5)
            w_acc[:] = 0.
            # ws to be used in AndersonAcceleration
            ws_intercept = np.append(ws, -1) if self.fit_intercept else ws

            if self.verbose:
                print(f'Iteration {t + 1}, {ws_size} feats in subpb.')

            # 2) do iterations on smaller problem
            is_sparse = sparse.issparse(X)
            for epoch in range(self.max_epochs):
                if is_sparse:
                    _cd_epoch_sparse(
                        X.data, X.indptr, X.indices, y, w[:n_features], Xw,
                        datafit, penalty, ws)
                else:
                    _cd_epoch(X, y, w[:n_features], Xw, datafit, penalty, ws)

                # update intercept
                if self.fit_intercept:
                    intercept_old = w[-1]
                    w[-1] -= datafit.intercept_update_step(y, Xw)
                    Xw += (w[-1] - intercept_old)

                # 3) do Anderson acceleration on smaller problem
                w_acc[ws_intercept], Xw_acc[:], is_extrap = accelerator.extrapolate(
                    w[ws_intercept], Xw)

                if is_extrap:  # avoid computing p_obj for un-extrapolated w, Xw
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
                    if self.ws_strategy == "subdiff":
                        opt_ws = penalty.subdiff_distance(w[:n_features], grad_ws, ws)
                    elif self.ws_strategy == "fixpoint":
                        opt_ws = dist_fix_point(
                            w[:n_features], grad_ws, datafit, penalty, ws)

                    stop_crit_in = np.max(opt_ws)
                    if max(self.verbose - 1, 0):
                        p_obj = (datafit.value(y, w[:n_features], Xw) +
                                 penalty.value(w[:n_features]))
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
            p_obj = datafit.value(y, w[:n_features], Xw) + penalty.value(w[:n_features])
            obj_out.append(p_obj)
        return w, np.array(obj_out), stop_crit

    def path(self, X, y, model, datafit, penalty, alphas=None, w_init=None,
             return_n_iter=False):
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
        coefs = np.zeros((n_features + model.fit_intercept, n_alphas), order='F',
                         dtype=X.dtype)
        stop_crits = np.zeros(n_alphas)
        p0 = self.p0

        if return_n_iter:
            n_iters = np.zeros(n_alphas, dtype=int)

        for t in range(n_alphas):
            alpha = alphas[t]
            penalty.alpha = alpha
            if self.verbose:
                to_print = "##### Computing alpha %d/%d" % (t + 1, n_alphas)
                print("#" * len(to_print))
                print(to_print)
                print("#" * len(to_print))
            if t > 0:
                w = coefs[:, t - 1].copy()
                # TODO tmp fix debug for L05:  p0 > replace by 1 (?)
                p0 = max(np.sum(penalty.generalized_support(w)), p0)
            else:
                if w_init is not None:
                    w = w_init.copy()
                    supp_size = penalty.generalized_support(w[:n_features]).sum()
                    p0 = max(supp_size, p0)
                    if supp_size:
                        Xw = X @ w[:n_features] + model.fit_intercept * w[-1]
                    # TODO explain/clean this hack
                    else:
                        Xw = np.zeros_like(y)
                else:
                    w = np.zeros(n_features + model.fit_intercept, dtype=X.dtype)
                    Xw = np.zeros(X.shape[0], dtype=X.dtype)

            sol = self.solve(X, y, datafit, penalty, w, Xw)

            coefs[:, t] = sol[0]
            stop_crits[t] = sol[-1]

            if return_n_iter:
                n_iters[t] = len(sol[1])

        results = alphas, coefs, stop_crits
        if return_n_iter:
            results += (n_iters,)
        return results


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
