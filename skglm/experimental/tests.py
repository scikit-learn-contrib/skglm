import numpy as np
from numpy.linalg import norm
from statsmodels.regression import linear_model

from skglm.utils import make_correlated_data
from skglm.experimental.sqrt_lasso import SqrtLasso


def test_alpha_max():
    n_samples, n_features = 50, 10
    X, y, _ = make_correlated_data(n_samples, n_features, random_state=0)

    alpha_max = norm(X.T @ y, ord=np.inf) / (np.sqrt(n_samples) * norm(y))

    sqrt_lasso = SqrtLasso(alpha=alpha_max, fit_intercept=False)
    sqrt_lasso.fit(X, y)

    np.testing.assert_equal(sqrt_lasso.coef_.flatten(), 0)


def test_vs_statsmodels():
    n_samples, n_features = 50, 10
    X, y, _ = make_correlated_data(n_samples, n_features, random_state=0)

    alpha_max = norm(X.T @ y, ord=np.inf) / (np.sqrt(n_samples) * norm(y))
    n_alphas = 3
    alphas = alpha_max * np.geomspace(1, 1e-2, n_alphas+1)[1:]

    sqrt_lasso = SqrtLasso()
    coefs_skglm = sqrt_lasso.path(X, y, alphas)[1]

    coefs_statsmodels = np.zeros((n_features, len(alphas)))

    # fit statsmodels on path
    for i in range(n_alphas):
        alpha = alphas[i]
        model = linear_model.OLS(y, X)
        model = model.fit_regularized(method='sqrt_lasso', L1_wt=1.,
                                      alpha=n_samples * alpha)
        coefs_statsmodels[:, i] = model.params

    np.testing.assert_almost_equal(coefs_skglm, coefs_statsmodels, decimal=4)


if __name__ == '__main__':
    pass
