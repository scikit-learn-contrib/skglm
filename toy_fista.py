import numpy as np
from numpy.linalg import norm
from skglm.datafits.single_task import Quadratic
from skglm.solvers import FISTA
from skglm.penalties import L1
from skglm.estimators import Lasso
from skglm.utils import make_correlated_data, compiled_clone


X, y, _ = make_correlated_data(n_samples=200, n_features=100, random_state=24)

n_samples, n_features = X.shape
alpha_max = norm(X.T @ y, ord=np.inf) / n_samples

alpha = alpha_max / 10

max_iter = 1000
obj_freq = 100
tol = 1e-10

solver = FISTA(max_iter=max_iter, tol=tol, opt_freq=obj_freq, verbose=1)
datafit = compiled_clone(Quadratic())
datafit.initialize(X, y)
penalty = compiled_clone(L1(alpha))
w = solver.solve(X, y, datafit, penalty)

clf = Lasso(alpha=alpha, tol=tol, fit_intercept=False)
clf.fit(X, y)

np.testing.assert_allclose(w, clf.coef_, rtol=1e-5)
