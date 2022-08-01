import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from skglm.penalties import L1
from skglm.utils import make_correlated_data, compiled_clone

from skglm.prototype_PN.log_datafit import Pr_LogisticRegression
from skglm.prototype_PN.pn_solver import pn_solver


def test_log_datafit():
    n_samples, n_features = 10, 20

    w = np.ones(n_features)
    X, y, _ = make_correlated_data(n_samples, n_features)
    y = np.sign(y)
    Xw = X @ w

    log_datafit = Pr_LogisticRegression()

    grad = log_datafit.raw_gradient(y, Xw)
    hess = log_datafit.raw_hessian(y, Xw)

    np.testing.assert_equal(grad.shape, (n_samples,))
    np.testing.assert_equal(hess.shape, (n_samples,))

    np.testing.assert_almost_equal(-grad * (y + len(y) * grad), hess)


def test_alpha_max():
    n_samples, n_features = 10, 20
    X, y, _ = make_correlated_data(n_samples, n_features)
    y = np.sign(y)

    alpha_max = np.linalg.norm(X.T @ y, ord=np.inf) / (2 * n_samples)

    log_datafit = compiled_clone(Pr_LogisticRegression())
    l1_penalty = compiled_clone(L1(alpha_max))
    w = pn_solver(X, y, log_datafit, l1_penalty)[0]

    np.testing.assert_equal(w, 0)


@pytest.mark.parametrize(('rho'), [1e-1, 1e-2])
def test_pn_vs_sklearn(rho):
    n_samples, n_features = 10, 20

    X, y, _ = make_correlated_data(n_samples, n_features, random_state=0)
    y = np.sign(y)

    alpha_max = np.linalg.norm(X.T @ y, ord=np.inf) / (2 * n_samples)
    alpha = rho * alpha_max

    sk_log_reg = LogisticRegression(penalty='l1', C=1/(n_samples * alpha),
                                    fit_intercept=False, tol=1e-9, solver='liblinear')
    sk_log_reg.fit(X, y)

    log_datafit = compiled_clone(Pr_LogisticRegression())
    l1_penalty = compiled_clone(L1(alpha))
    w = pn_solver(X, y, log_datafit, l1_penalty, tol=1e-9)[0]

    np.testing.assert_allclose(sk_log_reg.coef_ - w, 0, rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
    pass
