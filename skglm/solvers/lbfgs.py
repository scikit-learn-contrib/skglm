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

        def objective(w):
            Xw = X @ w
            datafit_value = datafit.value(y, w, Xw)
            penalty_value = penalty.value(w)

            return datafit_value + penalty_value

        def d_jac(w):
            Xw = X @ w
            datafit_grad = datafit.gradient(X, y, Xw)
            penalty_grad = penalty.gradient(w)

            return datafit_grad + penalty_grad

        def s_jac(w):
            Xw = X @ w
            datafit_grad = datafit.gradient_sparse(X.data, X.indptr, X.indices, y, Xw)
            penalty_grad = penalty.gradient(w)

            return datafit_grad + penalty_grad

        def callback_post_iter(w_k):
            # save p_obj
            p_obj = objective(w_k)
            p_objs_out.append(p_obj)

            if self.verbose:
                grad = jac(w_k)
                stop_crit = norm(grad, ord=np.inf)

                it = len(p_objs_out)
                print(
                    f"Iteration {it}: {p_obj:.10f}, "
                    f"stopping crit: {stop_crit:.2e}"
                )

        n_features = X.shape[1]
        w = np.zeros(n_features) if w_init is None else w_init
        jac = s_jac if issparse(X) else d_jac
        p_objs_out = []

        result = scipy.optimize.minimize(
            fun=objective,
            jac=jac,
            x0=w,
            method="L-BFGS-B",
            options=dict(
                maxiter=self.max_iter,
                gtol=self.tol,
                ftol=0.  # set ftol=0. to control convergence using only gtol
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
        # scipy LBFGS uses || projected gradient ||_oo to check convergence, cf. `gtol`
        # in https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html
        stop_crit = norm(result.jac, ord=np.inf)

        return w, np.asarray(p_objs_out), stop_crit
