import numpy as np
import scipy.optimize
import pytest

from sklearn.linear_model import HuberRegressor
from numpy.testing import assert_allclose, assert_array_less

from skglm.datafits import Huber, Logistic, Poisson, Gamma, Cox
from skglm.penalties import L1, WeightedL1
from skglm.solvers import AndersonCD, ProxNewton
from skglm import GeneralizedLinearEstimator
from skglm.utils.data import make_correlated_data
from skglm.utils.jit_compilation import compiled_clone


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


def test_gamma():
    try:
        import statsmodels.api as sm
    except ImportError:
        pytest.xfail("`statsmodels` not found. `Gamma` datafit can't be tested.")

    # When n_samples < n_features, the unregularized Gamma objective does not have a
    # unique minimizer.
    rho = 1e-2
    n_samples, n_features = 100, 10
    X, y, _ = make_correlated_data(n_samples, n_features, random_state=0)
    y[y <= 0] = 0.1
    tol = 1e-14

    alpha_max = np.linalg.norm(X.T @ (1 - y), ord=np.inf) / n_samples
    alpha = rho * alpha_max

    gamma_model = sm.GLM(y, X, family=sm.families.Gamma(sm.families.links.Log()))
    gamma_results = gamma_model.fit_regularized(
        method="elastic_net", L1_wt=1, cnvrg_tol=tol, alpha=alpha)

    clf = GeneralizedLinearEstimator(
        datafit=Gamma(),
        penalty=L1(alpha),
        solver=ProxNewton(fit_intercept=False, tol=tol)
    ).fit(X, y)

    np.testing.assert_allclose(clf.coef_, gamma_results.params, rtol=1e-6)


def test_cox():
    rng = np.random.RandomState(1265)
    n_samples, n_features = 10, 30

    # generate data
    X = rng.randn(n_samples, n_features)
    tm = rng.choice(n_samples*n_features, size=n_samples, replace=True).astype(float)
    s = rng.choice(2, size=n_samples).astype(float)
    y = (tm, s)

    # generate dummy w, Xw
    w = rng.randn(n_features)
    Xw = X @ w

    # check datafit
    cox_df = compiled_clone(Cox())

    cox_df.initialize(X, (tm, s))
    cox_df.value(y, w, Xw)

    # perform test 10 times to consider truncation errors
    # due to usage of finite differences to evaluate grad and Hessian
    for _ in range(10):

        # generate dummy w, Xw
        w = rng.randn(n_features)
        Xw = X @ w

        # check gradient
        np.testing.assert_allclose(
            scipy.optimize.check_grad(
                lambda x: cox_df.value(y, w, x),
                lambda x: cox_df.raw_grad(y, x),
                x0=Xw,
                seed=rng
            ),
            0., atol=1e-6
        )

        # check hessian upper bound
        # Hessian minus its upper bound must be negative semi definite
        hess_upper_bound = np.diag(cox_df.raw_hessian(y, Xw))
        hess = scipy.optimize.approx_fprime(
            xk=Xw,
            f=lambda x: cox_df.raw_grad(y, x),
        )

        positive_eig = np.linalg.eigh(hess - hess_upper_bound)[0]
        positive_eig = positive_eig[positive_eig >= 0.]

        np.testing.assert_allclose(positive_eig, 0., atol=1e-6)


if __name__ == '__main__':
    pass
