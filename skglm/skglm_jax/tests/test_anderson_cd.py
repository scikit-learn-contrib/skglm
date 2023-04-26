import numpy as np
from numpy.linalg import norm
from skglm.utils.data import make_correlated_data

from skglm.skglm_jax.anderson_cd import AndersonCD
from skglm.skglm_jax.datafits import QuadraticJax
from skglm.skglm_jax.penalties import L1Jax

from skglm.estimators import Lasso


def test_solver():
    n_samples, n_features = 100, 200
    random_state = 135

    X, y, _ = make_correlated_data(n_samples, n_features, random_state=random_state)

    lmbd_max = norm(X.T @ y, ord=np.inf) / n_samples
    lmbd = 1e-2 * lmbd_max

    datafit = QuadraticJax()
    penalty = L1Jax(lmbd)
    w = AndersonCD(max_iter=30, verbose=1, p0=2).solve(X, y, datafit, penalty)

    estimator = Lasso(alpha=lmbd, fit_intercept=False).fit(X, y)

    np.testing.assert_allclose(w, estimator.coef_, atol=1e-6)


if __name__ == "__main__":
    test_solver()
