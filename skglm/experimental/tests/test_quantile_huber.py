import pytest
import numpy as np
from numpy.testing import assert_allclose
from sklearn.linear_model import QuantileRegressor
from sklearn.datasets import make_regression
from skglm.experimental.quantile_huber import SmoothQuantileRegressor


@pytest.mark.parametrize('quantile', [0.3, 0.5, 0.7])
@pytest.mark.parametrize('fit_intercept', [True, False])
def test_quantile_huber_matches_sklearn(quantile, fit_intercept):
    """Test that SmoothQuantileRegressor with small delta matches sklearn's
    QuantileRegressor."""
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)

    sk_est = QuantileRegressor(quantile=quantile, alpha=0.1,
                               solver='highs', fit_intercept=fit_intercept).fit(X, y)
    smooth_est = SmoothQuantileRegressor(
        quantile=quantile,
        alpha=0.1,
        delta_init=0.5,
        delta_final=0.00001,
        n_deltas=15,
        verbose=False,
        fit_intercept=fit_intercept,
    ).fit(X, y)

    assert not np.allclose(sk_est.coef_, 0, atol=1e-8), (
        "All coefficients in sk_est are (near) zero: alpha may be too high.")
    assert not np.allclose(smooth_est.coef_, 0, atol=1e-8), (
        "All coefficients in smooth_est are (near) zero: alpha may be too high.")

    assert_allclose(smooth_est.coef_, sk_est.coef_, atol=1e-3)
    if fit_intercept:
        assert_allclose(smooth_est.intercept_, sk_est.intercept_, atol=1e-3)
