import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, KFold
from skglm.datafits import Quadratic
from skglm.penalties import L1_plus_L2
from skglm.solvers import AndersonCD
from skglm.cv import GeneralizedLinearEstimatorCV
import pytest


@pytest.mark.parametrize("n_samples,n_features,noise",
                         [(100, 10, 0.1), (100, 500, 0.2), (100, 500, 0.3)])
def test_elasticnet_cv_matches_sklearn(n_samples, n_features, noise):
    """Test GeneralizedLinearEstimatorCV matches sklearn GridSearchCV for ElasticNet."""
    seed = 42
    X, y = make_regression(n_samples=n_samples,
                           n_features=n_features, noise=noise, random_state=seed)

    n = X.shape[0]
    alpha_max = np.max(np.abs(X.T @ y)) / n
    alphas = alpha_max * np.array([1, 0.1, 0.01, 0.001])
    l1_ratios = np.array([0.2, 0.5, 0.8])
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)

    sklearn_model = GridSearchCV(
        ElasticNet(max_iter=10000, tol=1e-8),
        {'alpha': alphas, 'l1_ratio': l1_ratios},
        cv=cv, scoring='neg_mean_squared_error', n_jobs=1
    ).fit(X, y)

    skglm_model = GeneralizedLinearEstimatorCV(
        Quadratic(), L1_plus_L2(0.1, 0.5), AndersonCD(max_iter=10000, tol=1e-8),
        alphas=alphas, l1_ratio=l1_ratios, cv=5, random_state=seed, n_jobs=1
    ).fit(X, y)

    np.testing.assert_equal(sklearn_model.best_params_['alpha'],
                            skglm_model.alpha_)
    np.testing.assert_equal(sklearn_model.best_params_['l1_ratio'],
                            skglm_model.l1_ratio_)
    np.testing.assert_allclose(sklearn_model.best_estimator_.coef_,
                               skglm_model.coef_.ravel(), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(sklearn_model.best_estimator_.intercept_,
                               skglm_model.intercept_, rtol=1e-4, atol=1e-6)
