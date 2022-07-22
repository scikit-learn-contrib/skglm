import numpy as np
from numpy.linalg import norm
from scipy.sparse import issparse
# from numba.types import bool_
from numba.experimental import jitclass
from numba import float64
from numba.types import bool_
import time

from skglm.utils import ST

from sklearn.utils.validation import check_X_y
from sklearn.utils.multiclass import check_classification_targets

from sklearn.linear_model._base import LinearModel
from sklearn.preprocessing import LabelEncoder

from skglm.penalties.base import BasePenalty
from skglm.datafits.base import BaseDatafit
from skglm.solvers import cd_solver_path, bcd_solver_path


class QuadraticNoJit(BaseDatafit):
    def __init__(self):
        pass

    def initialize(self, X, y):
        self.Xty = X.T @ y
        n_features = X.shape[1]
        self.lipschitz = np.zeros(n_features, dtype=X.dtype)
        for j in range(n_features):
            self.lipschitz[j] = (X[:, j] ** 2).sum() / len(y)

    def initialize_sparse(
            self, X_data, X_indptr, X_indices, y):
        n_features = len(X_indptr) - 1
        self.Xty = np.zeros(n_features, dtype=X_data.dtype)
        self.lipschitz = np.zeros(n_features, dtype=X_data.dtype)
        for j in range(n_features):
            nrm2 = 0.
            xty = 0
            for idx in range(X_indptr[j], X_indptr[j + 1]):
                nrm2 += X_data[idx] ** 2
                xty += X_data[idx] * y[X_indices[idx]]

            self.lipschitz[j] = nrm2 / len(y)
            self.Xty[j] = xty

    def value(self, y, w, Xw):
        return np.sum((y - Xw) ** 2) / (2 * len(Xw))

    def gradient_scalar(self, X, y, w, Xw, j):
        return (X[:, j] @ Xw - self.Xty[j]) / len(Xw)

    def gradient_scalar_sparse(self, X_data, X_indptr, X_indices, y, Xw, j):
        XjTXw = 0.
        for i in range(X_indptr[j], X_indptr[j+1]):
            XjTXw += X_data[i] * Xw[X_indices[i]]
        return (XjTXw - self.Xty[j]) / len(Xw)

    def full_grad_sparse(
            self, X_data, X_indptr, X_indices, y, Xw):
        n_features = X_indptr.shape[0] - 1
        n_samples = y.shape[0]
        grad = np.zeros(n_features, dtype=Xw.dtype)
        for j in range(n_features):
            XjTXw = 0.
            for i in range(X_indptr[j], X_indptr[j + 1]):
                XjTXw += X_data[i] * Xw[X_indices[i]]
            grad[j] = (XjTXw - self.Xty[j]) / n_samples
        return grad


class L1NoJit(BasePenalty):
    """L1 penalty."""

    def __init__(self, alpha):
        self.alpha = alpha

    def value(self, w):
        """Compute L1 penalty value."""
        return self.alpha * np.sum(np.abs(w))

    def prox_1d(self, value, stepsize, j):
        """Compute proximal operator of the L1 penalty (soft-thresholding operator)."""
        return ST(value, self.alpha * stepsize)

    def subdiff_distance(self, w, grad, ws):
        """Compute distance of negative gradient to the subdifferential at w."""
        subdiff_dist = np.zeros_like(grad)
        for idx, j in enumerate(ws):
            if w[j] == 0:
                # distance of - grad_j to  [-alpha, alpha]
                subdiff_dist[idx] = max(0, np.abs(grad[idx]) - self.alpha)
            else:
                # distance of - grad_j to alpha * sign(w[j])
                subdiff_dist[idx] = np.abs(
                    - grad[idx] - np.sign(w[j]) * self.alpha)
        return subdiff_dist

    def is_penalized(self, n_features):
        """Return a binary mask with the penalized features."""
        # return np.ones(n_features, bool_)
        return np.ones(n_features, bool_)

    def generalized_support(self, w):
        """Return a mask with non-zero coefficients."""
        return w != 0

    def alpha_max(self, gradient0):
        """Return penalization value for which 0 is solution."""
        return np.max(np.abs(gradient0))


spec_quadratic = [
    ('Xty', float64[:]),
    ('lipschitz', float64[:]),
]


class GeneralizedLinearEstimator(LinearModel):
    def __init__(self, datafit=None, penalty=None, is_classif=False, max_iter=100,
                 max_epochs=50_000, p0=10, tol=1e-4, fit_intercept=True,
                 warm_start=False, ws_strategy="subdiff", verbose=0):
        super(GeneralizedLinearEstimator, self).__init__()
        self.is_classif = is_classif
        self.tol = tol
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.warm_start = warm_start
        self.verbose = verbose
        self.max_epochs = max_epochs
        self.p0 = p0
        self.ws_strategy = ws_strategy
        self.penalty = penalty
        self.datafit = datafit

    # TODO this is never used in fact
    def path(self, X, y, alphas, coef_init=None, return_n_iter=True, **params):
        penalty = jitclass(['alpha', float64])(self.penalty.__class__)()
        datafit = jitclass(spec_quadratic)(self.datafit.__class__)()

        path_func = cd_solver_path if y.ndim == 1 else bcd_solver_path
        return path_func(
            X, y, datafit, penalty, alphas=alphas,
            coef_init=coef_init, max_iter=self.max_iter,
            return_n_iter=return_n_iter, max_epochs=self.max_epochs, p0=self.p0,
            tol=self.tol, use_acc=True, ws_strategy=self.ws_strategy,
            verbose=self.verbose)

    def fit(self, X, y):

        if not hasattr(self, "n_features_in_"):
            self.n_features_in_ = X.shape[1]

        self.classes_ = None
        n_classes_ = 0

        if self.is_classif:
            check_classification_targets(y)
            enc = LabelEncoder()
            y = enc.fit_transform(y)
            self.classes_ = enc.classes_
            n_classes_ = len(self.classes_)

        check_X_params = dict(dtype=[np.float64, np.float32], order='F',
                              accept_sparse='csc', copy=self.fit_intercept)
        check_y_params = dict(ensure_2d=False, order='F')

        X, y = self._validate_data(X, y, validate_separately=(check_X_params,
                                                              check_y_params))
        is_sparse = issparse(X)

        n_samples = X.shape[0]
        if n_samples != y.shape[0]:
            raise ValueError("X and y have inconsistent dimensions (%d != %d)"
                             % (n_samples, y.shape[0]))

        # X, y, X_offset, y_offset, X_scale = _preprocess_data(
        #     X, y, self.fit_intercept, copy=False)

        if not self.warm_start or not hasattr(self, "coef_"):
            self.coef_ = None

        path_func = cd_solver_path if y.ndim == 1 else bcd_solver_path

        # TODO so far this takes time every time fit is called unfortunately
        penalty = jitclass([('alpha', float64)])(
            self.penalty.__class__)(self.penalty.alpha)
        datafit = jitclass(spec_quadratic)(self.datafit.__class__)()

        _, coefs, kkt = path_func(
            X, y, datafit, penalty, alphas=[self.penalty.alpha],
            coef_init=self.coef_, max_iter=self.max_iter,
            max_epochs=self.max_epochs, p0=self.p0, verbose=self.verbose,
            tol=self.tol, ws_strategy=self.ws_strategy)

        self.coef_, self.stop_crit_ = coefs[..., 0], kkt[-1]
        self.n_iter_ = len(kkt)
        # TODO: handle intercept for Quadratic, Logistic, etc.
        # self._set_intercept(X_offset, y_offset, X_scale)
        self.intercept_ = 0.

        return self

    def predict(self, X):
        if self.is_classif:
            scores = self._decision_function(X).ravel()
            if len(scores.shape) == 1:
                indices = (scores > 0).astype(int)
            else:
                indices = scores.argmax(axis=1)
            return self.classes_[indices]
        else:
            return self._decision_function(X)

    def get_params(self, deep=False):
        params = super().get_params(deep)
        filtered_types = (float, int, str, np.ndarray)
        penalty_params = [('penalty__', p, getattr(self.penalty, p)) for p in
                          dir(self.penalty) if p[0] != "_" and
                          type(getattr(self.penalty, p)) in filtered_types]
        datafit_params = [('datafit__', p, getattr(self.datafit, p)) for p in
                          dir(self.datafit) if p[0] != "_" and
                          type(getattr(self.datafit, p)) in filtered_types]
        for p_prefix, p_key, p_val in penalty_params + datafit_params:
            params[p_prefix + p_key] = p_val
        return params


if __name__ == "__main__":
    from benchopt.datasets.simulated import make_correlated_data
    X, y, _ = make_correlated_data(n_samples=1000, n_features=2000, random_state=0)
    alpha = norm(X.T @ y, ord=np.inf) / len(y) / 10

    penalty = L1NoJit(alpha)
    datafit = QuadraticNoJit()

    clf = GeneralizedLinearEstimator(datafit, penalty, verbose=0)

    t0 = time.time()
    clf.fit(X, y)
    t1 = time.time()
    print(f"first call: {t1 - t0:.3f} s")

    t0 = time.time()
    clf.fit(X, y)
    t1 = time.time()
    print(f"second call: {t1 - t0:.3f} s")
