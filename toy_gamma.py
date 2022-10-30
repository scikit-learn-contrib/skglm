import numpy as np
from numpy.linalg import norm
from skglm.utils import make_correlated_data
from skglm.penalties import L1
from skglm.datafits import Gamma
from skglm.solvers import ProxNewton
from skglm.estimators import GeneralizedLinearEstimator
# from sklearn.linear_model import GammaRegressor
import statsmodels.api as sm


n_samples = 100
n_features = 40
X, y, _ = make_correlated_data(n_samples, n_features, random_state=0)
y[y <= 0] = 0.1
alpha_max = norm(X.T @ y, ord=np.inf) / n_samples
alpha = alpha_max * 0.01 
alpha = 0

tol = 1e-12

# clf_sk = GammaRegressor(alpha=alpha * n_samples, fit_intercept=False).fit(X, y)

gamma_model = sm.GLM(y, X, family=sm.families.Gamma(sm.families.links.Log()))
gamma_results = gamma_model.fit_regularized(
    method="elastic_net", L1_wt=1, cnvrg_tol=tol, alpha=alpha * n_samples)

clf = GeneralizedLinearEstimator(
    datafit=Gamma(),
    penalty=L1(alpha),
    solver=ProxNewton(fit_intercept=False, verbose=2, tol=tol)
).fit(X, y)


np.testing.assert_allclose(clf.coef_, gamma_results.params, rtol=1e-5)

# print(norm(clf.coef_ - clf_sk.coef_))
# print("sklearn:", (clf_sk.coef_[:10]))
# print("skglm:", (clf.coef_))
# print("sm:", (gamma_results.params))

# np.testing.assert_allclose(clf_sk.coef_, gamma_results.params, rtol=1e-5)
# np.testing.assert_allclose(clf.coef_, gamma_results.params, rtol=1e-5)
