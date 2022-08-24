import pytest
from itertools import product

import numpy as np
from numpy.linalg import norm
from sklearn.linear_model import Lasso

from skglm.penalties import L1
from skglm.solvers.gram import gram_solver
from skglm.utils import make_correlated_data, compiled_clone


@pytest.mark.parametrize("n_samples, n_features",
                         product([100, 200], [50, 90]))
def test_alpha_max(n_samples, n_features):
    X, y, _ = make_correlated_data(n_samples, n_features, random_state=0)
    alpha_max = norm(X.T @ y, ord=np.inf) / n_samples

    l1_penalty = compiled_clone(L1(alpha_max))
    w = gram_solver(X, y, l1_penalty, tol=1e-9, verbose=2)[0]

    np.testing.assert_equal(w, 0)


@pytest.mark.parametrize("n_samples, n_features, rho",
                         product([50, 100], [20, 80], [1e-1, 1e-2]))
def test_vs_lasso_sklearn(n_samples, n_features, rho):
    X, y, _ = make_correlated_data(n_samples, n_features, random_state=0)
    alpha_max = norm(X.T @ y, ord=np.inf) / n_samples
    alpha = rho * alpha_max

    sk_lasso = Lasso(alpha, fit_intercept=False, tol=1e-9)
    sk_lasso.fit(X, y)

    l1_penalty = compiled_clone(L1(alpha))
    w = gram_solver(X, y, l1_penalty, tol=1e-9, verbose=0, p0=10)[0]

    print(
        f"skglm:   {compute_obj(X, y, alpha, w)}\n"
        f"sklearn: {compute_obj(X, y, alpha, sk_lasso.coef_.flatten())}"
    )

    # np.testing.assert_allclose(w, sk_lasso.coef_.flatten(), rtol=1e-5, atol=1e-5)


def compute_obj(X, y, alpha, coef):
    return norm(y - X @ coef) ** 2 / (2 * len(y)) + alpha * norm(coef, ord=1)


if __name__ == '__main__':
    test_vs_lasso_sklearn(50, 80, 0.01)

    # print(
    #     f"skglm:   {compute_obj(X, y, alpha, w)}\n"
    #     f"sklearn: {compute_obj(X, y, alpha, sk_lasso.coef_.flatten())}"
    # )
    pass
