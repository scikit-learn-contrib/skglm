import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from skglm.utils import make_correlated_data
from skglm.experimental import ReweightedEstimator
from skglm.solvers import AndersonCD


n_samples, n_features = 200, 500
X, y, _ = make_correlated_data(n_samples=n_samples, n_features=n_features, random_state=24)
alpha_max = norm(X.T @ y, ord=np.inf) / n_samples

alpha = alpha_max / 10

solver = AndersonCD(tol=1e-10, fit_intercept=False, warm_start=True, verbose=2)
clf = ReweightedEstimator(alpha, solver=solver)
clf.fit(X, y)

# reweighting can't increase the L0.5 objective
assert clf.loss_history_[0] > clf.loss_history_[-1]
diffs = np.diff(clf.loss_history_)
np.testing.assert_array_less(diffs, 1e-5)

print(clf.loss_history_)

