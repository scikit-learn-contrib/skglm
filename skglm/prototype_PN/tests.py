import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from skglm.penalties import L1
from skglm.utils import make_correlated_data, compiled_clone

from skglm.datafits import Logistic
from skglm.prototype_PN.pn_PAB import prox_newton_solver

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
    hess = log_datafit.raw_hessian(y, Xw, grad)

    np.testing.assert_equal(grad.shape, (n_samples,))
    np.testing.assert_equal(hess.shape, (n_samples,))

    exp_yXw = np.exp(-y * Xw)
    np.testing.assert_almost_equal(-y * exp_yXw / (1 + exp_yXw) / len(y), grad)
    np.testing.assert_almost_equal(exp_yXw / (1 + exp_yXw) ** 2 / len(y), hess)


@pytest.mark.parametrize('X_density', [1, 0.5])
def test_alpha_max(X_density):
    n_samples, n_features = 10, 20
    X, y, _ = make_correlated_data(n_samples, n_features, X_density=X_density)
    y = np.sign(y)

    alpha_max = np.linalg.norm(X.T @ y, ord=np.inf) / (2 * n_samples)

    log_datafit = compiled_clone(Pr_LogisticRegression())
    l1_penalty = compiled_clone(L1(alpha_max))
    w = pn_solver(X, y, log_datafit, l1_penalty)[0]

    np.testing.assert_equal(w, 0)


@pytest.mark.parametrize("rho, X_density", [[1e-1, 1], [1e-2, 0.5]])
def test_pn_vs_sklearn(rho, X_density):
    n_samples, n_features = 10, 20

    X, y, _ = make_correlated_data(n_samples, n_features, random_state=0,
                                   X_density=X_density)
    y = np.sign(y)

    alpha_max = np.linalg.norm(X.T @ y, ord=np.inf) / (2 * n_samples)
    alpha = rho * alpha_max

    sk_log_reg = LogisticRegression(penalty='l1', C=1/(n_samples * alpha),
                                    fit_intercept=False, tol=1e-9, solver='liblinear')
    sk_log_reg.fit(X, y)

    log_datafit = compiled_clone(Pr_LogisticRegression())
    l1_penalty = compiled_clone(L1(alpha))
    w = pn_solver(X, y, log_datafit, l1_penalty, tol=1e-9)[0]

    np.testing.assert_allclose(sk_log_reg.coef_ - w, 0, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("rho, X_density", [[1e-1, 1], [1e-2, 0.5]])
def test_PN_PAB_vs_sklearn(rho, X_density):
    n_samples, n_features = 10, 20

    X, y, _ = make_correlated_data(n_samples, n_features, random_state=0, X_density=X_density)
    y = np.sign(y)
    tol = 1e-9

    alpha_max = np.linalg.norm(X.T @ y, ord=np.inf) / (2 * n_samples)
    alpha = rho * alpha_max

    datafit = Logistic()
    pen = L1(alpha=alpha)

    pen = compiled_clone(pen)
    datafit = compiled_clone(datafit)

    w = np.zeros(n_features)
    Xw = np.zeros(n_samples)
    w_newton = prox_newton_solver(X, y, datafit, pen, w, Xw, tol=tol)[0]

    estimator_sk = LogisticRegression(
        C=1/(alpha * n_samples), fit_intercept=False, tol=tol, penalty='l1',
        solver='liblinear')
    estimator_sk.fit(X, y)

    np.testing.assert_allclose(w_newton, np.ravel(estimator_sk.coef_), atol=1e-5)


if __name__ == '__main__':
    pass
