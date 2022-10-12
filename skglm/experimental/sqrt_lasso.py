import warnings
import numpy as np
from numba import njit
from numpy.linalg import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model._base import LinearModel, RegressorMixin

from skglm.penalties import L1
from skglm.utils import compiled_clone, ST, ST_vec, proj_L2ball
from skglm.datafits.base import BaseDatafit
from skglm.solvers.prox_newton import ProxNewton


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

        Raises
        ------
            Exception
                if value of residuals is less than ``1e-2 * ||y||``.
        """
        minus_residual = Xw - y
        norm_residuals = norm(minus_residual)

        if norm_residuals < 1e-2 * norm(y):
            raise ValueError("SmallResidualException")

        return minus_residual / (norm_residuals * np.sqrt(len(y)))

    def raw_hessian(self, y, Xw):
        """Diagonal matrix upper bounding the Hessian."""
        n_samples = len(y)
        fill_value = 1 / (np.sqrt(n_samples) * norm(y - Xw))
        return np.full(n_samples, fill_value)


class SqrtLasso(LinearModel, RegressorMixin):
    """Square root Lasso estimator based on Prox Newton solver.

    The optimization objective for square root Lasso is::

        |y - X w||_2 / sqrt(n_samples) + alpha * ||w||_1

    Parameters
    ----------
    alpha : float, default 1
        Penalty strength.

    max_iter : int, default 20
        Maximum number of outer iterations.

    max_pn_iter : int, default 1000
        Maximum number of prox Newton iterations on each subproblem.

    p0 : int, default 10
        Minimum number of features to be included in the working set.

    tol : float, default 1e-4
        Tolerance for convergence.

    verbose : bool, default False
        Amount of verbosity. 0/False is silent.
    """

    def __init__(self, alpha=1., max_iter=100, max_pn_iter=100, p0=10,
                 tol=1e-4, verbose=0):
        super().__init__()
        self.alpha = alpha
        self.max_iter = max_iter
        self.max_pn_iter = max_pn_iter

        self.p0 = p0
        self.tol = tol
        self.verbose = verbose

    def fit(self, X, y):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array or sparse CSC matrix, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self :
            Fitted estimator.
        """
        self.coef_ = self.path(X, y, alphas=[self.alpha])[1][0]
        self.intercept_ = 0.  # TODO handle fit_intercept
        return self

    def path(self, X, y, alphas=None, eps=1e-3, n_alphas=10):
        """Compute Lasso path.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Design matrix.

        y : array, shape (n_samples,)
            Target vector.

        alphas : array, shape (n_alphas,) default None
            Grid of alpha. If None a path is constructed from
            (0, alpha_max] with a length ``eps``.

        eps: float, default 1e-2
            Length of the path. ``eps=1e-3`` means that
            ``alpha_min = 1e-3 * alpha_max``.

        n_alphas: int, default 10
            Number of alphas along the path. This argument is
            ignored if ``alphas`` was provided.

        Returns
        -------
        alphas : array, shape (n_alphas,)
            The alphas along the path where models are computed.

        coefs : array, shape (n_features, n_alphas)
            Coefficients along the path.
        """
        if not hasattr(self, "solver_"):
            self.solver_ = ProxNewton(
                tol=self.tol, max_iter=self.max_iter, verbose=self.verbose)
        # build path
        if alphas is None:
            alpha_max = norm(X.T @ y, ord=np.inf) / (np.sqrt(len(y)) * norm(y))
            alphas = alpha_max * np.geomspace(1, eps, n_alphas)
        else:
            n_alphas = len(alphas)
            alphas = np.sort(alphas)[::-1]

        n_features = X.shape[1]
        sqrt_quadratic = compiled_clone(SqrtQuadratic())
        l1_penalty = compiled_clone(L1(1.))  # alpha is set along the path

        coefs = np.zeros((n_alphas, n_features))

        for i in range(n_alphas):
            if self.verbose:
                to_print = "##### Computing alpha %d/%d" % (i + 1, n_alphas)
                print("#" * len(to_print))
                print(to_print)
                print("#" * len(to_print))

            l1_penalty.alpha = alphas[i]
            # no warm start for the first alpha
            coef_init = coefs[i].copy() if i else np.zeros(n_features)

            try:
                coef, _, _ = self.solver_.solve(
                    X, y, sqrt_quadratic, l1_penalty,
                    w_init=coef_init, Xw_init=X @ coef_init)
                coefs[i] = coef
            except ValueError as val_exception:
                # make sure to catch residual error
                # it's implemented this way as Numba doesn't support custom Exception
                if not str(val_exception) == "SmallResidualException":
                    raise

                # save coef despite not converging
                # coef_init holds a ref to coef
                coef = coef_init
                res_norm = norm(y - X @ coef)
                warnings.warn(
                    f"Small residuals prevented the solver from converging "
                    f"at alpha={alphas[i]:.2e} (residuals' norm: {res_norm:.4e}). "
                    "Consider fitting with higher alpha.",
                    ConvergenceWarning
                )
                coefs[i] = coef
                break

        return alphas, coefs


def _chambolle_pock_sqrt(X, y, alpha, max_iter=1000, obj_freq=10, verbose=False):
    """Apply Chambolle-Pock algorithm to solve square-root Lasso.

    The objective function is:
        min_w ||Xw - y||_2/sqrt(n_samples) + alpha * ||w||_1.
    """
    n_samples, n_features = X.shape
    # dual variable is z, primal is w
    z_old = np.zeros(n_samples)
    z = z_old.copy()
    w = np.zeros(n_features)

    objs = []

    L = norm(X, ord=2)
    # take primal and dual stepsizes equal
    tau = 0.99 / L
    sigma = 0.99 / L

    for t in range(max_iter):
        w = ST_vec(w - tau * X.T @ (2 * z - z_old), alpha * np.sqrt(n_samples) * tau)
        z_old = z.copy()
        z[:] = proj_L2ball(z + sigma * (X @ w - y))

        if t % obj_freq == 0:
            objs.append(norm(X @ w - y) / np.sqrt(n_samples) + alpha * norm(w, ord=1))
            if verbose:
                print("Iter {0}, obj {1:.10f}".format(t, objs[-1]))

    return w, z, objs


@njit
def _fercoq_bianchi_sqrt(X, y, alpha, max_iter=1000, obj_freq=10, verbose=False):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    z = np.zeros(n_samples)
    z_bar = np.zeros(n_samples)
    Xw = np.zeros(n_samples)
    sigma = 1 / n_features
    taus = 1 / (2 * np.sum(X ** 2, axis=0))
    objs = []

    for it in range(max_iter):
        # roughly equal cost to CP? rather 3 n_samples * n_features
        for j in range(n_features):
            z_bar[:] = proj_L2ball(z + sigma * (Xw - y))  # n_samples
            w_j = w[j]
            w[j] = ST(w[j] - taus[j] * X[:, j] @ (2 * z_bar - z),
                      alpha * np.sqrt(n_samples) * taus[j])  # n_samples
            update = w[j] - w_j
            if update != 0:
                Xw += X[:, j] * update  # n_samples
            z *= (1 - 1 / n_features)
            z += z_bar / n_features

        if it % obj_freq == 0:
            objs.append(norm(Xw - y) / np.sqrt(n_samples) + alpha * norm(w, ord=1))
            if verbose:
                # print("Iter %d, obj %f" % (it, objs[-1]))
                print("Iter ", it, objs[-1])

    return w, z, objs
