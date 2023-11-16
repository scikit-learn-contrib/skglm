import numpy as np
from scipy.sparse import issparse
from skglm.solvers.base import BaseSolver
from skglm.solvers.common import construct_grad, construct_grad_sparse
from skglm.utils.prox_funcs import _prox_vec
from skglm.utils.validation import check_obj_solver_attr


class FISTA(BaseSolver):
    r"""ISTA solver with Nesterov acceleration (FISTA).

    Attributes
    ----------
    max_iter : int, default 100
        Maximum number of iterations.

    tol : float, default 1e-4
        Tolerance for convergence.

    verbose : bool, default False
        Amount of verbosity. 0/False is silent.

    References
    ----------
    .. [1] Beck, A. and Teboulle M.
           "A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse
           problems", 2009, SIAM J. Imaging Sci.
           https://epubs.siam.org/doi/10.1137/080716542
    """

    _datafit_required_attr = ("get_global_lipschitz", ("gradient", "gradient_scalar"))
    _penalty_required_attr = (("prox_1d", "prox_vec"),)

    def __init__(self, max_iter=100, tol=1e-4, opt_strategy="subdiff", verbose=0):
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.opt_strategy = opt_strategy
        self.fit_intercept = False   # needed to be passed to GeneralizedLinearEstimator
        self.warm_start = False

    def solve(self, X, y, datafit, penalty, w_init=None, Xw_init=None):
        p_objs_out = []
        n_samples, n_features = X.shape
        all_features = np.arange(n_features)
        X_is_sparse = issparse(X)
        t_new = 1.

        w = w_init.copy() if w_init is not None else np.zeros(n_features)
        z = w_init.copy() if w_init is not None else np.zeros(n_features)
        Xw = Xw_init.copy() if Xw_init is not None else np.zeros(n_samples)

        if X_is_sparse:
            lipschitz = datafit.get_global_lipschitz_sparse(
                X.data, X.indptr, X.indices, y
            )
        else:
            lipschitz = datafit.get_global_lipschitz(X, y)

        for n_iter in range(self.max_iter):
            t_old = t_new
            t_new = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2
            w_old = w.copy()

            if X_is_sparse:
                if hasattr(datafit, "gradient_sparse"):
                    grad = datafit.gradient_sparse(
                        X.data, X.indptr, X.indices, y, X @ z)
                else:
                    grad = construct_grad_sparse(
                        X.data, X.indptr, X.indices, y, z, X @ z, datafit, all_features)
            else:
                if hasattr(datafit, "gradient"):
                    grad = datafit.gradient(X, y, X @ z)
                else:
                    grad = construct_grad(X, y, z, X @ z, datafit, all_features)

            step = 1 / lipschitz
            z -= step * grad
            if hasattr(penalty, "prox_vec"):
                w = penalty.prox_vec(z, step)
            else:
                w = _prox_vec(w, z, penalty, step)
            Xw = X @ w
            z = w + (t_old - 1.) / t_new * (w - w_old)

            if self.opt_strategy == "subdiff":
                opt = penalty.subdiff_distance(w, grad, all_features)
            elif self.opt_strategy == "fixpoint":
                opt = np.abs(w - penalty.prox_vec(w - grad / lipschitz, 1 / lipschitz))
            else:
                raise ValueError(
                    "Unknown error optimality strategy. Expected "
                    f"`subdiff` or `fixpoint`. Got {self.opt_strategy}")

            stop_crit = np.max(opt)

            p_obj = datafit.value(y, w, Xw) + penalty.value(w)
            p_objs_out.append(p_obj)
            if self.verbose:
                print(
                    f"Iteration {n_iter+1}: {p_obj:.10f}, "
                    f"stopping crit: {stop_crit:.2e}"
                )

            if stop_crit < self.tol:
                if self.verbose:
                    print(f"Stopping criterion max violation: {stop_crit:.2e}")
                break
        return w, np.array(p_objs_out), stop_crit

    def custom_compatibility_check(self, X, y, datafit, penalty):
        # check datafit support sparse data
        check_obj_solver_attr(
            datafit, solver=self,
            required_attr=self._datafit_required_attr,
            support_sparse=issparse(X)
        )
