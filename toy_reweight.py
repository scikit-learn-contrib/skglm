import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from skglm.utils import make_correlated_data
from skglm.experimental import ReweightedLasso

n_samples, n_features = 200, 500
X, y, _ = make_correlated_data(n_samples=n_samples, n_features=n_features, random_state=24)
alpha_max = norm(X.T @ y, ord=np.inf) / n_samples

alpha = alpha_max / 10

clf = ReweightedLasso(alpha=alpha, verbose=2, tol=1e-10)
clf.fit(X, y)

# reweighting can't increase the L0.5 objective
assert clf.loss_history_[0] > clf.loss_history_[-1]
diffs = np.diff(clf.loss_history_)
np.testing.assert_array_less(diffs, 1e-5)

print(clf.loss_history_)

