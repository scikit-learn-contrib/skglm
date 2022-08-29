import warnings
import numpy as np
from numpy.linalg import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model._base import LinearModel, RegressorMixin

from skglm.penalties import L1
from skglm.utils import compiled_clone
from skglm.datafits.base import BaseDatafit
from skglm.solvers.prox_newton import prox_newton


class SqrtQuadratic(BaseDatafit):
    """Square root quadratic datafit.

    The datafit reads::
        ||y - Xw||_2 / sqrt(n_samples) 
    """

    def __init__(self):
        pass

    def get_spec(self):
        spec = ()
        return spec

    def params_to_dict(self):
        return dict()

    def value(self, y, w, Xw):
        return np.linalg.norm(y - Xw) / np.sqrt(len(y))

    def raw_grad(self, y, Xw):
        """Compute gradient of datafit w.r.t ``Xw``.

        Raises:
        -------
            Exception
                if value of residuals is too small (less than ``1e-10``).
        """
        minus_residual = Xw - y
        norm_residuals = norm(minus_residual)

        if norm_residuals < 1e-10:
            raise Exception(
                f"Too small residuals will impact the convergence of the solver."
            )
        return minus_residual / norm_residuals / np.sqrt(len(y))

    def raw_hessian(self, y, Xw):
        """Upper bound of the hessian w.r.t eigenvalues."""
        n_samples = len(y)
        fill_value = 1 / (np.sqrt(n_samples) * norm(y - Xw))
        return np.full(n_samples, fill_value)


class SqrtLasso(LinearModel, RegressorMixin):

    def __init__(self, alpha=1., max_iter=100, max_pn_iter=1000, p0=10,
                 tol=1e-4, verbose=0):
        super().__init__()
        self.alpha = alpha
        self.max_iter = max_iter
        self.max_pn_iter = max_pn_iter

        self.p0 = p0
        self.tol = tol
        self.verbose = verbose

    def fit(self, X, y):
        self.coef_ = self.path(X, y, alphas=[self.alpha])[1][0]
        self.intercept_ = 0.  # TODO handle fit_intercept
        return self

    def path(self, X, y, alphas=None):
        n_features = X.shape[1]
        n_alphas = len(alphas)
        alphas = np.sort(alphas)[::-1]

        sqrt_quadratic = compiled_clone(SqrtQuadratic())
        l1_penalty = compiled_clone(L1(1.))

        coefs = np.zeros((n_alphas, n_features))
        stop_criteria = np.zeros(n_alphas)
        n_iters = np.zeros(n_alphas)

        for i in range(n_alphas):
            if self.verbose:
                print(f"======== alpha {i+1} ========")

            l1_penalty.alpha = alphas[i]
            # no warm start for the first alpha
            coef_init = coefs[i].copy() if i else np.zeros(n_features)

            try:
                coef, p_objs_out, stop_crit = prox_newton(
                    X, y, sqrt_quadratic, l1_penalty,
                    w_init=coef_init, max_iter=self.max_iter,
                    max_pn_iter=self.max_pn_iter,
                    tol=self.tol, verbose=self.verbose)
            except:
                # save coef despite not converging
                # coef_init holds a ref to coef
                coef = coef_init
                residual = sqrt_quadratic.value(y, coef, X @ coef)
                warnings.warn(
                    f"Small residuals will prevent the solver from converging.\n"
                    f"Consider taking alphas greater than {alphas[i]:.4e}\n"
                    f"value of residual: {residual:.4e}",
                    ConvergenceWarning
                )

            coefs[i] = coef
            stop_criteria[i] = stop_crit
            n_iters[i] = len(p_objs_out)

        return alphas, coefs, stop_criteria, n_iters
