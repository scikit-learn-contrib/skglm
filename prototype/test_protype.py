import pytest

import numpy as np
from numpy.linalg import norm

from sklearn.linear_model import Lasso
from skglm.experimental import SqrtLasso
from skglm.utils import make_correlated_data

from prototype.pd_lasso import cp_lasso, fb_lasso, forward_backward, cd
from prototype.pd_sqrt_lasso import fb_sqrt_lasso, cp_sqrt_lasso


@pytest.mark.parametrize('solver', [fb_lasso, cp_lasso, forward_backward, cd])
def test_solver(solver):
    rho = 0.1
    n_samples, n_features = 50, 10
    A, b, _ = make_correlated_data(n_samples, n_features,
                                   random_state=0)

    alpha_max = norm(A.T @ b, ord=np.inf)
    alpha = rho * alpha_max

    w, _ = solver(A, b, alpha, max_iter=1000)
    lasso = Lasso(fit_intercept=False,
                  alpha=alpha / n_samples).fit(A, b)

    np.testing.assert_allclose(w, lasso.coef_.flatten(), atol=1e-4)


@pytest.mark.parametrize('solver', [fb_sqrt_lasso, cp_sqrt_lasso])
def test_solver(solver):
    rho = 0.1
    n_samples, n_features = 50, 10
    A, b, _ = make_correlated_data(n_samples, n_features,
                                   random_state=0)

    alpha_max = norm(A.T @ b, ord=np.inf) / norm(b)
    alpha = rho * alpha_max

    w, _ = solver(A, b, alpha, max_iter=1000)
    lasso = SqrtLasso(alpha=alpha / np.sqrt(n_samples), tol=1e-9).fit(A, b)

    np.testing.assert_allclose(w, lasso.coef_.flatten(), atol=1e-4)


if __name__ == '__main__':
    test_solver(fb_sqrt_lasso)
    pass