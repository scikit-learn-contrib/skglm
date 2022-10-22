import numpy as np
from numpy.linalg import norm
from sklearn.metrics import f1_score

from skglm.penalties.separable import L0_5
from skglm.utils import make_correlated_data
from skglm.estimators import GeneralizedLinearEstimator
from skglm.experimental import IterativeReweightedL1
from skglm.solvers import AndersonCD


n_samples, n_features = 200, 500
X, y, w_true = make_correlated_data(
    n_samples=n_samples, n_features=n_features, random_state=24)

alpha_max = norm(X.T @ y, ord=np.inf) / n_samples
alpha = alpha_max / 100
tol = 1e-10


def _obj(w):
    return (np.sum((y - X @ w) ** 2) / (2 * n_samples)
            + alpha * np.sum(np.sqrt(np.abs(w))))


iterative_l05 = IterativeReweightedL1(
    penalty=L0_5(alpha),
    solver=AndersonCD(tol=tol, fit_intercept=False))

# `subdiff` strategy for WS is uninformative for L0_5
direct_l05 = GeneralizedLinearEstimator(
    penalty=L0_5(alpha),
    solver=AndersonCD(tol=tol, fit_intercept=False, ws_strategy="fixpoint"))

iterative_l05.fit(X, y)

# reweighting can't increase the L0.5 objective
assert iterative_l05.loss_history_[0] > iterative_l05.loss_history_[-1]
diffs = np.diff(iterative_l05.loss_history_)
np.testing.assert_array_less(diffs, 1e-5)

direct_l05.fit(X, y)

print("#" * 20)
print("Objective value")
print("Reweighting:", _obj(iterative_l05.coef_))
print("Direct prox:", _obj(direct_l05.coef_))

print("#" * 20)

print("Support recovery (F1 score)")
print("Reweighting:", f1_score(w_true != 0, iterative_l05.coef_ != 0))
print("Direct prox:", f1_score(w_true != 0, direct_l05.coef_ != 0))
