import numpy as np

from benchopt.datasets.simulated import make_correlated_data
from sklearn.linear_model import HuberRegressor


from skglm import GeneralizedLinearEstimator
from skglm.datafits import Huber
from skglm.penalties import WeightedL1

X, y, _ = make_correlated_data(n_samples=20, n_features=10, random_state=0)


# no L2 regularization (alpha=0)
clf = HuberRegressor(fit_intercept=False, epsilon=1.35, alpha=0, tol=1e-10).fit(X, y)


print(clf.coef_)

delta = clf.epsilon * clf.scale_

our = GeneralizedLinearEstimator(
    datafit=Huber(delta),
    penalty=WeightedL1(1, np.zeros(X.shape[1])),
    is_classif=False,
    verbose=2,
    tol=1e-10).fit(X, y)
