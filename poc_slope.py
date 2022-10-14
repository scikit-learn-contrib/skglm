from numba import njit, float64
import numpy as np
from numpy.linalg import norm
from skglm.solvers import FISTA
from skglm.datafits import Quadratic
from skglm.penalties import BasePenalty
from skglm.estimators import Lasso
from skglm.utils import make_correlated_data, compiled_clone


X, y, _ = make_correlated_data(n_samples=200, n_features=100, random_state=24)

n_samples, n_features = X.shape
alpha_max = norm(X.T @ y, ord=np.inf) / n_samples

alpha = np.repeat(alpha_max / 10, n_features)

max_iter = 1000
obj_freq = 100
tol = 1e-10


class SLOPE(BasePenalty):
    def __init__(self, alphas):
        self.alphas = alphas
    
    def get_spec(self):
        spec = (
            ('alphas', float64[:]),
        )
        return spec
    
    def params_to_dict(self):
        return dict(alphas=self.alphas)

    def value(self, w):
        sorted_indices = np.argsort(w)[::-1]  # descending order
        return np.sum(w * self.alphas[sorted_indices])

    def prox_vec(self, x, stepsize):
        def _prox(_x, _alphas):
            sign_x = np.sign(_x)
            _x = np.abs(_x)
            sorted_indices = np.argsort(_x)[::-1]
            prox = fast_prox_SL1(_x[sorted_indices], _alphas)
            prox[sorted_indices] = prox
            return prox * sign_x
        return _prox(x, self.alphas * stepsize)


@njit
def fast_prox_SL1(y, alphas):
    # w, alphas: nonnegative and nonincreasing sequences
    n_features = y.shape[0]
    x = np.empty(n_features)

    k = 0
    idx_i = np.empty((n_features,), dtype=np.int64)
    idx_j = np.empty((n_features,), dtype=np.int64)
    s = np.empty((n_features,), dtype=np.float64)
    w = np.empty((n_features,), dtype=np.float64)

    for i in range(n_features):
        idx_i[k] = i
        idx_j[k] = i
        s[k] = y[i] - alphas[i]
        w[k] = s[k]

        while k > 0 and w[k - 1] <= w[k]:
            k -= 1
            idx_j[k] = i
            s[k] += s[k+1]
            w[k] = s[k] / (i - idx_i[k] + 1)

        k += 1

    for j in range(k):
        d = w[j]
        d = 0 if d < 0 else d
        for i in range(idx_i[j], idx_j[j] + 1):
            x[i] = d

    return x



solver = FISTA(max_iter=max_iter, tol=tol, opt_freq=obj_freq, verbose=1)
penalty = compiled_clone(SLOPE(alpha))
datafit = compiled_clone(Quadratic())
datafit.initialize(X, y)
w = solver.solve(X, y, datafit, penalty)


# check that solution is equal to Lasso's
estimator = Lasso(alpha[0], fit_intercept=False, tol=tol)
estimator.fit(X, y)

np.testing.assert_allclose(w, estimator.coef_, rtol=1e-5)
