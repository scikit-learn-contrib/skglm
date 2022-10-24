import numpy as np
from numpy.linalg import norm

from skglm.penalties.separable import L0_5
from skglm.utils import make_correlated_data
from skglm.experimental import IterativeReweightedL1
from skglm.solvers import AndersonCD


n_samples, n_features = 20, 50
X, y, w_true = make_correlated_data(
    n_samples=n_samples, n_features=n_features, random_state=24)

alpha_max = norm(X.T @ y, ord=np.inf) / n_samples
alpha = alpha_max / 100
tol = 1e-10


def test_decreasing_loss():
    # reweighting can't increase the L0.5 objective
    iterative_l05 = IterativeReweightedL1(
        penalty=L0_5(alpha),
        solver=AndersonCD(tol=tol, fit_intercept=False)).fit(X, y)
    np.testing.assert_array_less(
        iterative_l05.loss_history_[-1], iterative_l05.loss_history_[0])
    diffs = np.diff(iterative_l05.loss_history_)
    np.testing.assert_array_less(diffs, 1e-5)
