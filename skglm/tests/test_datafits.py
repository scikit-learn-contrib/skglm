import numpy as np

from sklearn.linear_model import HuberRegressor
from numpy.testing import assert_allclose

from skglm.datafits import Huber
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
