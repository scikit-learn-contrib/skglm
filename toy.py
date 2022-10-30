import numpy as np
from numpy.linalg import norm
from sklearn.linear_model import Lasso as Lasso_sk
from sklearn.linear_model import ElasticNet as ElasticNet_sk
from skglm.estimators import Lasso, ElasticNet
from skglm.utils import make_correlated_data

n_samples = 30
n_features = 70
X, y, w_true =  make_correlated_data(n_samples, n_features, random_state=0)

alpha_max = norm(X.T @ y, ord=np.inf) / n_samples
alpha = alpha_max * 0.1
l1_ratio = 0.3

tol = 1e-12

clf = Lasso(
    alpha, positive=True, tol=tol, fit_intercept=False, ws_strategy="subdiff",
    verbose=2).fit(X, y)

clf_sk = Lasso_sk(alpha, positive=True, tol=tol, fit_intercept=False).fit(X, y)

# enet = ElasticNet(
#     alpha, l1_ratio, positive=True, tol=tol, fit_intercept=False,
#     ws_strategy="fixpoint", verbose=0).fit(X, y)
#
# enet_sk = ElasticNet_sk(
#     alpha=alpha, l1_ratio=l1_ratio, positive=True, tol=tol, fit_intercept=False).fit(X, y)


np.testing.assert_allclose(clf.coef_, clf_sk.coef_, rtol=1e-5)
# np.testing.assert_allclose(enet.coef_, enet_sk.coef_, rtol=1e-5)

