import pytest

import numpy as np
from numpy.linalg import norm
from skglm.utils.data import make_correlated_data

from skglm.skglm_jax.anderson_cd import AndersonCD
from skglm.skglm_jax.fista import Fista
from skglm.skglm_jax.datafits import QuadraticJax
from skglm.skglm_jax.penalties import L1Jax

from skglm.estimators import Lasso


@pytest.mark.parametrize(
    "solver", [AndersonCD(),
               Fista(use_auto_diff=True),
               Fista(use_auto_diff=False)])
def test_solver(solver):
    random_state = 135
    n_samples, n_features = 10_000, 100

    X, y, _ = make_correlated_data(n_samples, n_features, random_state=random_state)

    lmbd_max = norm(X.T @ y, ord=np.inf) / n_samples
    lmbd = 1e-2 * lmbd_max

    datafit = QuadraticJax()
    penalty = L1Jax(lmbd)
    w = solver.solve(X, y, datafit, penalty)

    estimator = Lasso(alpha=lmbd, fit_intercept=False).fit(X, y)

    np.testing.assert_allclose(w, estimator.coef_, atol=1e-4)


if __name__ == "__main__":
    import time

    start = time.perf_counter()
    test_solver(AndersonCD(verbose=2))
    end = time.perf_counter()

    print("Elapsed time:", end - start)
    pass
