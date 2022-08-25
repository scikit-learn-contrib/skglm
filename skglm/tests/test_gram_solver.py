import pytest
from itertools import product

import numpy as np
from numpy.linalg import norm
from sklearn.linear_model import Lasso

from skglm.penalties import L1
from skglm.solvers.gram_cd import gram_cd_solver
from skglm.utils import make_correlated_data, compiled_clone


@pytest.mark.parametrize("n_samples, n_features, X_density",
                         product([100, 200], [50, 90], [1., 0.6]))
def test_alpha_max(n_samples, n_features, X_density):
    X, y, _ = make_correlated_data(n_samples, n_features,
                                   random_state=0, X_density=X_density)
    alpha_max = norm(X.T @ y, ord=np.inf) / n_samples

    l1_penalty = compiled_clone(L1(alpha_max))
    w = gram_cd_solver(X, y, l1_penalty, tol=1e-9, verbose=0)[0]

    np.testing.assert_equal(w, 0)


@pytest.mark.parametrize("n_samples, n_features, rho, X_density",
                         product([500, 100], [30, 80], [1e-1, 1e-2, 1e-3], [1., 0.8]))
def test_vs_lasso_sklearn(n_samples, n_features, rho, X_density):
    X, y, _ = make_correlated_data(n_samples, n_features,
                                   random_state=0, X_density=X_density)
    alpha_max = norm(X.T @ y, ord=np.inf) / n_samples
    alpha = rho * alpha_max

    sk_lasso = Lasso(alpha, fit_intercept=False, tol=1e-9)
    sk_lasso.fit(X, y)

    l1_penalty = compiled_clone(L1(alpha))
    w = gram_cd_solver(X, y, l1_penalty, tol=1e-9, verbose=0, max_iter=1000)[0]

    np.testing.assert_allclose(w, sk_lasso.coef_.flatten(), rtol=1e-7, atol=1e-7)


if __name__ == '__main__':
    test_vs_lasso_sklearn(100, 10, 0.01)
    pass
