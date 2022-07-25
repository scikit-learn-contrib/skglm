import numpy as np
from numpy.linalg import norm
# from numba.types import bool_

import time

from skglm import GeneralizedLinearEstimator
from skglm.penalties import L1
from skglm.datafits import Quadratic
from skglm.datafits.base import jit_cached_compile


if __name__ == "__main__":
    from benchopt.datasets.simulated import make_correlated_data
    X, y, _ = make_correlated_data(n_samples=1000, n_features=2000, random_state=0)
    alpha = norm(X.T @ y, ord=np.inf) / len(y) / 10

    penalty = L1(alpha)
    datafit = Quadratic()

    # penalty_jit = jit_cached_compile(
    #     penalty.__class__,
    #     penalty.get_spec(),
    # )(**penalty.params_to_dict())

    # datafit_jit = jit_cached_compile(
    #     datafit.__class__,
    #     datafit.get_spec(),
    #     to_float32=X.dtype is np.float32,
    # )(**datafit.params_to_dict())

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
