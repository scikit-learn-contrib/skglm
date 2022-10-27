import numpy as np
from numpy.linalg import norm
from skglm.utils import make_correlated_data
from skglm.penalties import L1
from skglm.datafits import Gamma
from skglm.solvers import ProxNewton
from skglm.estimators import GeneralizedLinearEstimator
from sklearn.linear_model import GammaRegressor


n_samples = 10
n_features = 30
X, y, _ = make_correlated_data(n_samples, n_features, random_state=0)
y[y <= 0] = 0.001
alpha_max = norm(X.T @ y, ord=np.inf) / n_samples 
alpha = alpha_max * 0.05
alpha = 0

tol = 1e-12

clf = GeneralizedLinearEstimator(
    datafit=Gamma(),
    penalty=L1(alpha),
    solver=ProxNewton(fit_intercept=False, verbose=2, tol=tol)
).fit(X, y)

clf_sk = GammaRegressor(alpha=alpha, fit_intercept=False, tol=1e-12).fit(X, y)

np.testing.assert_allclose(clf.coef_, clf_sk.coef_, rtol=1e-5)
