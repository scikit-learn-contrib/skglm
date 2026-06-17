import pytest
import numpy as np
from numpy.linalg import norm

from skglm.penalties import L1
from skglm import GeneralizedLinearEstimator
from skglm.experimental.pdcd_ws import PDCD_WS
from skglm.experimental.quantile_regression import Pinball

from skglm.utils.data import make_correlated_data
from sklearn.linear_model import QuantileRegressor


@pytest.mark.parametrize('quantile_level,n_samples,n_features',
                         ([[0.3, 50, 20], [0.5, 1000, 11], [0.7, 50, 100]])
                         )
def test_PDCD_WS(quantile_level, n_samples, n_features):
    X, y, _ = make_correlated_data(n_samples, n_features, random_state=123)

    # optimality condition for w = 0.
    #   for all g in subdiff pinball(y), g must be in subdiff ||.||_1(0)
    # hint: use max(x, 0) = (x + |x|) / 2 to get subdiff pinball
    alpha_max = norm(X.T @ (np.sign(y)/2 + (quantile_level - 0.5)), ord=np.inf)
    alpha = alpha_max / 5

    datafit = Pinball(quantile_level)
    penalty = L1(alpha)

    w = PDCD_WS(tol=1e-9).solve(X, y, datafit, penalty)[0]

    clf = QuantileRegressor(
        quantile=quantile_level,
        alpha=alpha/n_samples,
        fit_intercept=False,
        solver='highs',
    ).fit(X, y)

    np.testing.assert_allclose(w, clf.coef_, atol=1e-5)
    # unrelated: test compatibility when inside GLM:
    estimator = GeneralizedLinearEstimator(
        datafit=Pinball(.2),
        penalty=L1(alpha=1.),
        solver=PDCD_WS(),
    )
    estimator.fit(X, y)


if __name__ == '__main__':
    pass
