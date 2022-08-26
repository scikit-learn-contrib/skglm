import numpy as np
from numpy.linalg import norm
from sklearn.linear_model import Lasso

from skglm.penalties import L1
from skglm.utils import compiled_clone
from skglm.datafits.base import BaseDatafit
from skglm.solvers.prox_newton import prox_newton


class SqrtQuadratic(BaseDatafit):
    """norm(y - Xw) / sqrt(len(y))."""

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
        minus_residual = Xw - y
        return minus_residual / norm(minus_residual) / np.sqrt(len(y))

    def raw_hessian(self, y, Xw):
        n_samples = len(y)
        fill_value = 1 / (np.sqrt(n_samples) * norm(y - Xw))
        return np.full(n_samples, fill_value)


class SqrtLasso(Lasso):

    def __init__(self, alpha=1., max_iter=100, max_pn_iter=1000, p0=10,
                 tol=1e-4, fit_intercept=True, verbose=0,):
        super(Lasso, self).__init__(
            alpha=alpha, tol=tol, max_iter=max_iter,
            fit_intercept=fit_intercept)
        self.p0 = p0
        self.verbose = verbose
        self.max_pn_iter = max_pn_iter

    def path(self, X, y, alphas=None, **kwargs):
        n_features = X.shape[1]
        n_alphas = len(alphas)
        alphas = np.sort(alphas)[::-1]

        sqrt_quadratic = compiled_clone(SqrtQuadratic())
        l1_penalty = compiled_clone(L1(1.))

        coefs = np.zeros((n_features, n_alphas+1))
        stop_criteria = np.zeros(n_alphas)
        n_iters = np.zeros(n_alphas)

        for i in range(n_alphas):
            l1_penalty.alpha = alphas[i]
            coef_init = coefs[:, i-1]

            coef, p_objs_out, stop_crit = prox_newton(
                X, y, sqrt_quadratic, l1_penalty,
                w_init=coef_init, max_iter=self.max_iter,
                max_pn_iter=self.max_pn_iter,
                tol=self.tol, verbose=self.verbose)

            coefs[:, i] = coef
            stop_criteria[i] = stop_crit
            n_iters[i] = len(p_objs_out)

        return alphas, coefs[:, :-1], stop_criteria, n_iters
