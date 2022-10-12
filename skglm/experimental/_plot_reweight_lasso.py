import time
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
alpha = alpha_max / 10

def obj(w):
    return (np.sum((y - X @ w) ** 2) / (2 * n_samples)
            + alpha * np.sum(np.sqrt(np.abs(w))))

solver = AndersonCD(tol=1e-10, fit_intercept=False, warm_start=True, verbose=0)
iterative_l05 = IterativeReweightedL1(alpha, solver=solver)
direct_l05 = GeneralizedLinearEstimator(penalty=L0_5(alpha), solver=solver)

# TODO: cache compilation

print("#" * 20)
print("Time")

start = time.time()
iterative_l05.fit(X, y)
print("Reweighting:", time.time() - start)

# reweighting can't increase the L0.5 objective
assert iterative_l05.loss_history_[0] > iterative_l05.loss_history_[-1]
diffs = np.diff(iterative_l05.loss_history_)
np.testing.assert_array_less(diffs, 1e-5)

start = time.time()
direct_l05.fit(X, y)
print("Direct prox:", time.time() - start)

print("#" * 20)

# non-convex, not necessary same solution
print("Coefficient norms")
print("Reweighting:", norm(iterative_l05.coef_))
print("Direct:", norm(direct_l05.coef_))

print("#" * 20)
print("Objective value")
print("Reweighting:", obj(iterative_l05.coef_))
print("Direct:", obj(direct_l05.coef_))

print("#" * 20)

print("Support recovery (F1 score)")
print("Reweighting:", f1_score(w_true != 0, iterative_l05.coef_ != 0))
print("Direct:", f1_score(w_true != 0, direct_l05.coef_ != 0))

