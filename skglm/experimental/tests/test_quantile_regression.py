import pytest
import numpy as np
from numpy.linalg import norm

from skglm.penalties import L1
from skglm.experimental.pdcd_ws import PDCD_WS
from skglm.experimental.quantile_regression import Pinball

from skglm.utils.data import make_correlated_data
from sklearn.linear_model import QuantileRegressor


@pytest.mark.parametrize('quantile', [0.3, 0.5, 0.7])
def test_PDCD_WS(quantile):
    n_samples, n_features = 50, 10
    X, y, _ = make_correlated_data(n_samples, n_features, random_state=123)

    # optimality condition for w = 0.
    #   for all g in subdiff pinball(y), g must be in subdiff ||.||_1(0)
    # hint: use max(x, 0) = (x + |x|) / 2 to get subdiff pinball
    alpha_max = norm(X.T @ (np.sign(y)/2 + (quantile - 0.5)), ord=np.inf)
    alpha = alpha_max / 5

    w = PDCD_WS(
        dual_init=np.sign(y)/2 + (quantile - 0.5)
    ).solve(X, y, Pinball(quantile), L1(alpha))[0]

    clf = QuantileRegressor(
        quantile=quantile,
        alpha=alpha/n_samples,
        fit_intercept=False
    ).fit(X, y)

    np.testing.assert_allclose(w, clf.coef_, atol=1e-5)


if __name__ == '__main__':
    pass
