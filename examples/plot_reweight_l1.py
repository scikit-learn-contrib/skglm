"""
=================================================================
Timing comparison between direct prox computation and reweighting
=================================================================
Compare time and objective value of L0_5-regularized problem with
direct proximal computation and iterative reweighting.
"""
# Author: Pierre-Antoine Bannier<pierreantoine.bannier@gmail.com>

import time
import numpy as np
from numpy.linalg import norm

from skglm.penalties.separable import L0_5
from skglm.utils import make_correlated_data
from skglm.estimators import GeneralizedLinearEstimator
from skglm.experimental import IterativeReweightedL1
from skglm.solvers import AndersonCD


n_samples, n_features = 200, 500
X, y, w_true = make_correlated_data(
    n_samples=n_samples, n_features=n_features, random_state=24)

alpha_max = norm(X.T @ y, ord=np.inf) / n_samples
alphas = [alpha_max / 10, alpha_max / 100, alpha_max / 1000]
tol = 1e-10


def _obj(w):
    return (np.sum((y - X @ w) ** 2) / (2 * n_samples)
            + alpha * np.sum(np.sqrt(np.abs(w))))


def fit_l05(max_iter, alpha):
    start = time.time()
    iterative_l05 = IterativeReweightedL1(
        penalty=L0_5(alpha),
        solver=AndersonCD(tol=tol, max_iter=max_iter, fit_intercept=False)).fit(X, y)
    iterative_time = time.time() - start

    # `subdiff` strategy for WS is uninformative for L0_5
    start = time.time()
    direct_l05 = GeneralizedLinearEstimator(
        penalty=L0_5(alpha),
        solver=AndersonCD(tol=tol, max_iter=max_iter, fit_intercept=False,
                          ws_strategy="fixpoint")).fit(X, y)
    direct_time = time.time() - start

    results = {
        "iterative": (iterative_l05, iterative_time),
        "direct": (direct_l05, direct_time),
    }
    return results


# caching Numba compilation
fit_l05(1, alpha_max)

# actual run
for alpha in alphas:
    print("#" * 20)
    print(f"alpha = {alpha}")
    print("#" * 20)

    results = fit_l05(max_iter=100, alpha=alpha)
    iterative_l05, iterative_time = results["iterative"]
    direct_l05, direct_time = results["direct"]

    print("#" * 5)
    print("Time")
    print("Reweighting:", iterative_time)
    print("Direct prox:", direct_time)

    print("#" * 5)
    print("Objective value")
    print("Reweighting:", _obj(iterative_l05.coef_))
    print("Direct prox:", _obj(direct_l05.coef_))

    print("\n")
