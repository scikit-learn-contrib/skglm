import numpy as np

from skglm.penalties import L1_1
from skglm.datafits import QuadraticMultiTask
from skglm.estimators import GeneralizedLinearEstimator
from skglm.solvers import MultiTaskBCD
from skglm.utils import make_correlated_data

from sklearn.linear_model import Lasso


n_samples = 30
n_features = 50
n_tasks = 20


X, Y, _ = make_correlated_data(
    n_samples, n_features, n_tasks, density=0.2, random_state=0)

alpha_max = np.max(X.T @ Y) / n_samples
alpha = alpha_max * 0.01
tol = 1e-8

sk_clf = Lasso(alpha, fit_intercept=False, tol=tol).fit(X, Y)

ours_clf = GeneralizedLinearEstimator(
    datafit=QuadraticMultiTask(),
    penalty=L1_1(alpha),
    solver=MultiTaskBCD(
        tol=tol, fit_intercept=False, max_iter=1000, verbose=2, ws_strategy="subdiff")
).fit(X, Y)

np.testing.assert_allclose(sk_clf.coef_.T, ours_clf.coef_, rtol=1e-3)
