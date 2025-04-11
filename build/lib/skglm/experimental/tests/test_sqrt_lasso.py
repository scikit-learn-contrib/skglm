import pytest
import numpy as np
from numpy.linalg import norm

from skglm.penalties import L1
from skglm.utils.data import make_correlated_data
from skglm.experimental.sqrt_lasso import (SqrtLasso, SqrtQuadratic,
                                           _chambolle_pock_sqrt)
from skglm.experimental.pdcd_ws import PDCD_WS
from skglm.utils.jit_compilation import compiled_clone


def test_alpha_max():
    n_samples, n_features = 50, 10
    X, y, _ = make_correlated_data(n_samples, n_features, random_state=0)
    alpha_max = norm(X.T @ y, ord=np.inf) / norm(y)

    sqrt_lasso = SqrtLasso(alpha=alpha_max).fit(X, y)

    np.testing.assert_equal(sqrt_lasso.coef_, 0)


def test_vs_statsmodels():
    try:
        from statsmodels.regression import linear_model  # noqa
    except ImportError:
        pytest.xfail("This test requires statsmodels to run.")
    n_samples, n_features = 50, 10
    X, y, _ = make_correlated_data(n_samples, n_features, random_state=0)

    alpha_max = norm(X.T @ y, ord=np.inf) / norm(y)
    n_alphas = 3
    alphas = alpha_max * np.geomspace(1, 1e-2, n_alphas+1)[1:]

    sqrt_lasso = SqrtLasso(tol=1e-9)
    coefs_skglm = sqrt_lasso.path(X, y, alphas)[1]

    coefs_statsmodels = np.zeros((len(alphas), n_features))

    # fit statsmodels on path
    for i in range(n_alphas):
        alpha = alphas[i]
        # statsmodels solves: ||y - Xw||_2 + alpha * ||w||_1 / sqrt(n_samples)
        model = linear_model.OLS(y, X)
        model = model.fit_regularized(method='sqrt_lasso', L1_wt=1.,
                                      alpha=np.sqrt(n_samples) * alpha)
        coefs_statsmodels[i] = model.params

    np.testing.assert_almost_equal(coefs_skglm, coefs_statsmodels, decimal=4)


def test_prox_newton_cp():
    n_samples, n_features = 50, 10
    X, y, _ = make_correlated_data(n_samples, n_features, random_state=0)

    alpha_max = norm(X.T @ y, ord=np.inf) / norm(y)
    alpha = alpha_max / 10
    clf = SqrtLasso(alpha=alpha, tol=1e-12).fit(X, y)
    w, _, _ = _chambolle_pock_sqrt(X, y, alpha, max_iter=1000)
    np.testing.assert_allclose(clf.coef_, w)


@pytest.mark.parametrize('with_dual_init', [True, False])
def test_PDCD_WS(with_dual_init):
    n_samples, n_features = 50, 10
    X, y, _ = make_correlated_data(n_samples, n_features, random_state=0)

    alpha_max = norm(X.T @ y, ord=np.inf) / norm(y)
    alpha = alpha_max / 10

    dual_init = y / norm(y) if with_dual_init else None

    datafit = compiled_clone(SqrtQuadratic())
    penalty = compiled_clone(L1(alpha))

    w = PDCD_WS(dual_init=dual_init).solve(X, y, datafit, penalty)[0]
    clf = SqrtLasso(alpha=alpha, tol=1e-12).fit(X, y)
    np.testing.assert_allclose(clf.coef_, w, atol=1e-6)


if __name__ == '__main__':
    pass
