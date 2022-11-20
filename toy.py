import numpy as np
from sklearn.linear_model import LinearRegression

from skglm.datafits import Quadratic
from skglm.penalties import NNLS
from skglm.estimators import GeneralizedLinearEstimator
from skglm.solvers import AndersonCD
from skglm.utils import make_correlated_data

n_samples = 100
n_features = 40
X, y, _ = make_correlated_data(n_samples, n_features, density=0.2, random_state=0)

tol = 1e-8

clf = GeneralizedLinearEstimator(
    datafit=Quadratic(),
    penalty=NNLS(),
    solver=AndersonCD(tol=tol, fit_intercept=False, verbose=2),
).fit(X, y)

reg_nnls = LinearRegression(positive=True, fit_intercept=False)
reg_nnls.fit(X, y)

np.testing.assert_allclose(clf.coef_, reg_nnls.coef_, rtol=1e-3)
