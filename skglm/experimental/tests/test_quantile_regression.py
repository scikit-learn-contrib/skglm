import pytest
import numpy as np
from numpy.linalg import norm

from skglm.penalties import L1
from skglm.experimental.pdcd_ws import PDCD_WS
from skglm.experimental.quantile_regression import Pinball

from skglm.utils.data import make_correlated_data
from sklearn.linear_model import QuantileRegressor


@pytest.mark.parametrize('with_dual_init', [True, False])
def test_PDCD_WS(with_dual_init, quantile=0.5):
    n_samples, n_features = 50, 10
    X, y, _ = make_correlated_data(n_samples, n_features, random_state=123)

    alpha_max = norm(X.T @ (np.sign(y)/2 + (quantile - 0.5)), ord=np.inf)
    alpha = alpha_max / 10

    dual_init = np.sign(y)/2 + (quantile - 0.5) if with_dual_init else None

    w = PDCD_WS(
        dual_init=dual_init,
        verbose=1
    ).solve(X, y, Pinball(quantile), L1(alpha))[0]

    clf = QuantileRegressor(
        quantile=quantile,
        alpha=alpha/n_samples,
        fit_intercept=False
    ).fit(X, y)

    # np.testing.assert_allclose(w, clf.coef_, atol=1e-6)

    print(
        pin_ball_loss(y, X @ clf.coef_) + norm(clf.coef_, ord=1)
    )

    print(
        pin_ball_loss(y, X @ w) + norm(w, ord=1)
    )


# taken from benchopt
def pin_ball_loss(y_true, y_pred, quantile=0.5):
    diff = y_true - y_pred
    sign = (diff >= 0).astype(diff.dtype)
    loss = quantile * sign * diff - (1 - quantile) * (1 - sign) * diff
    return np.sum(loss)


if __name__ == '__main__':
    test_PDCD_WS(False, quantile=0.4)
