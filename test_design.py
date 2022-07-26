import numpy as np
from numpy.linalg import norm

import time

from skglm import GeneralizedLinearEstimator
from skglm.penalties import L1
from skglm.datafits import Quadratic


if __name__ == "__main__":
    from benchopt.datasets.simulated import make_correlated_data
    X, y, _ = make_correlated_data(n_samples=1000, n_features=2000, random_state=0)
    alpha = norm(X.T @ y, ord=np.inf) / len(y) / 10

    penalty = L1(alpha)
    datafit = Quadratic()

    clf = GeneralizedLinearEstimator(datafit, penalty, verbose=0)

    t0 = time.time()
    clf.fit(X, y)
    t1 = time.time()
    print(f"first call to fit with compilation: {t1 - t0:.3f} s")

    t0 = time.time()
    clf = GeneralizedLinearEstimator(datafit, penalty, verbose=0)
    clf.fit(X, y)
    t1 = time.time()
    # should not be so high
    print(f"second call to fit (compilation?): {t1 - t0:.3f} s")
