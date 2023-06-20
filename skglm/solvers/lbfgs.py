import warnings
from sklearn.exceptions import ConvergenceWarning

import numpy as np
import scipy.optimize
from numpy.linalg import norm
from scipy.sparse import issparse

from skglm.solvers import BaseSolver


class LBFGS(BaseSolver):
    """A wrapper for scipy L-BFGS solver.

    Refer to `scipy L-BFGS-B <https://docs.scipy.org/doc/scipy/reference/optimize.
    minimize-lbfgsb.html#optimize-minimize-lbfgsb>`_ documentation for details.

    Parameters
    ----------
    max_iter : int, default 20
        Maximum number of iterations.

    tol : float, default 1e-4
        Tolerance for convergence.

    verbose : bool, default False
        Amount of verbosity. 0/False is silent.
    """

    def __init__(self, max_iter=50, tol=1e-4, verbose=False):
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    def solve(self, X, y, datafit, penalty, w_init=None, Xw_init=None):

        def objective_function(w):
            Xw = X @ w
            datafit_value = datafit.value(y, w, Xw)
            penalty_value = penalty.value(w)

            return datafit_value + penalty_value

        def d_jacobian_function(w):
            Xw = X @ w
            datafit_grad = datafit.gradient(X, y, Xw)
            penalty_grad = penalty.gradient(w)

            return datafit_grad + penalty_grad

        def s_jacobian_function(w):
            Xw = X @ w
            datafit_grad = datafit.gradient_sparse(X.data, X.indptr, X.indices, y, Xw)
            penalty_grad = penalty.gradient(w)

            return datafit_grad + penalty_grad

        def callback_post_iter(w_k):
            # save p_obj
            p_obj = objective_function(w_k)
            p_objs_out.append(p_obj)

            if self.verbose:
                grad = jacobian_function(w_k)
                stop_crit = norm(grad)

                it = len(p_objs_out)
                print(
                    f"Iteration {it}: {p_obj:.10f}, "
                    f"stopping crit: {stop_crit:.2e}"
                )

        n_features = X.shape[1]
        w = np.zeros(n_features) if w_init is None else w_init
        jacobian_function = s_jacobian_function if issparse(X) else d_jacobian_function
        p_objs_out = []

        result = scipy.optimize.minimize(
            fun=objective_function,
            jac=jacobian_function,
            x0=w,
            method="L-BFGS-B",
            options=dict(
                maxiter=self.max_iter,
                gtol=self.tol
            ),
            callback=callback_post_iter,
        )

        if not result.success:
            warnings.warn(
                f"`LBFGS` did not converge for tol={self.tol:.3e} "
                f"and max_iter={self.max_iter}.\n"
                "Consider increasing `max_iter` and/or `tol`.",
                category=ConvergenceWarning
            )

        w = result.x
        stop_crit = norm(result.jac)

        return w, np.asarray(p_objs_out), stop_crit
