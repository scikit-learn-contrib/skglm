import numpy as np

from sklearn.linear_model import HuberRegressor
from numpy.testing import assert_allclose, assert_array_less

from skglm.datafits import Huber, Logistic
from skglm.penalties import WeightedL1
from skglm import GeneralizedLinearEstimator
from skglm.utils import make_correlated_data


def test_huber_datafit():
    # test only datafit: there does not exist other implems with sparse penalty
    X, y, _ = make_correlated_data(n_samples=20, n_features=10, random_state=0)
    # disable L2^2 regularization (alpha=0)
    their = HuberRegressor(
        fit_intercept=False, alpha=0, tol=1e-12, epsilon=1.35
    ).fit(X, y)

    # sklearn optimizes over a scale, we must match delta:
    delta = their.epsilon * their.scale_

    # TODO we should have an unpenalized solver
    ours = GeneralizedLinearEstimator(
        datafit=Huber(delta),
        penalty=WeightedL1(1, np.zeros(X.shape[1])),
        tol=1e-14,
    ).fit(X, y)

    assert_allclose(ours.coef_, their.coef_, rtol=1e-3)
    assert_array_less(ours.stop_crit_, ours.tol)


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

    exp_yXw = np.exp(-y * Xw)
    np.testing.assert_almost_equal(exp_yXw / (1 + exp_yXw) ** 2 / len(y), hess)
    np.testing.assert_almost_equal(-grad * (y + n_samples * grad), hess)


if __name__ == '__main__':
    pass
