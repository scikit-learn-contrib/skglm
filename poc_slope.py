import numpy as np
from numpy.linalg import norm
from skglm.solvers import FISTA
from skglm.datafits import Quadratic
from skglm.penalties import SLOPE
from skglm.estimators import Lasso
from skglm.utils import make_correlated_data, compiled_clone


X, y, _ = make_correlated_data(n_samples=200, n_features=100, random_state=24)

n_samples, n_features = X.shape
alpha_max = norm(X.T @ y, ord=np.inf) / n_samples

alpha = np.repeat(alpha_max / 10, n_features)

max_iter = 1000
obj_freq = 100
tol = 1e-10

solver = FISTA(max_iter=max_iter, tol=tol, verbose=1)
penalty = compiled_clone(SLOPE(alpha))
datafit = compiled_clone(Quadratic())
datafit.initialize(X, y)
w = solver.solve(X, y, datafit, penalty)


# check that solution is equal to Lasso's
estimator = Lasso(alpha[0], fit_intercept=False, tol=tol)
estimator.fit(X, y)

np.testing.assert_allclose(w, estimator.coef_, rtol=1e-5)
