import numpy as np
import pytest

from sklearn.linear_model import HuberRegressor
from numpy.testing import assert_allclose, assert_array_less

from skglm.datafits import Huber, Logistic, Poisson
from skglm.penalties import L1, WeightedL1
from skglm.solvers import AndersonCD, ProxNewton
from skglm import GeneralizedLinearEstimator
from skglm.utils import make_correlated_data


@pytest.mark.parametrize('fit_intercept', [False, True])
def test_huber_datafit(fit_intercept):
    # test only datafit: there does not exist other implems with sparse penalty
    X, y, _ = make_correlated_data(n_samples=20, n_features=10, random_state=0)
    # disable L2^2 regularization (alpha=0)
    their = HuberRegressor(
        fit_intercept=fit_intercept, alpha=0, tol=1e-12, epsilon=1.35
    ).fit(X, y)

    # sklearn optimizes over a scale, we must match delta:
    delta = their.epsilon * their.scale_

    # TODO we should have an unpenalized solver
    ours = GeneralizedLinearEstimator(
        datafit=Huber(delta),
        penalty=WeightedL1(1, np.zeros(X.shape[1])),
        solver=AndersonCD(tol=1e-14, fit_intercept=fit_intercept),
    ).fit(X, y)

    assert_allclose(ours.coef_, their.coef_, rtol=1e-3)
    assert_allclose(ours.intercept_, their.intercept_, rtol=1e-4)
    assert_array_less(ours.stop_crit_, ours.solver.tol)


def test_log_datafit():
    n_samples, n_features = 10, 20

    w = np.ones(n_features)
    X, y, _ = make_correlated_data(n_samples, n_features)
    y = np.sign(y)
    Xw = X @ w

    log_datafit = Logistic()
    grad = log_datafit.raw_grad(y, Xw)
    hess = log_datafit.raw_hessian(y, Xw)

    np.testing.assert_equal(grad.shape, (n_samples,))
    np.testing.assert_equal(hess.shape, (n_samples,))

    exp_minus_yXw = np.exp(-y * Xw)
    np.testing.assert_almost_equal(
        exp_minus_yXw / (1 + exp_minus_yXw) ** 2 / len(y), hess)
    np.testing.assert_almost_equal(-grad * (y + n_samples * grad), hess)


def test_poisson():
    try:
        from statsmodels.discrete.discrete_model import Poisson as PoissonRegressor  # noqa
    except ImportError:
        pytest.xfail("`statsmodels` not found. `Poisson` datafit can't be tested.")

    n_samples, n_features = 10, 22
    tol = 1e-14
    X, y, _ = make_correlated_data(n_samples, n_features, random_state=0)
    y = np.abs(y)

    alpha_max = np.linalg.norm(X.T @ (np.ones(n_samples) - y), ord=np.inf) / n_samples
    alpha = alpha_max * 0.1

    df = Poisson()
    pen = L1(alpha)

    solver = ProxNewton(tol=tol, fit_intercept=False)
    model = GeneralizedLinearEstimator(df, pen, solver).fit(X, y)

    poisson_regressor = PoissonRegressor(y, X, offset=None)
    res = poisson_regressor.fit_regularized(
        method="l1", size_trim_tol=tol, alpha=alpha * n_samples, trim_mode="size")
    w_statsmodels = res.params

    assert_allclose(model.coef_, w_statsmodels, rtol=1e-4)


if __name__ == '__main__':
    pass
