import numpy as np
from numpy.linalg import norm
from skglm.penalties import L1
from skglm.datafits.single_task import SqrtQuadratic
from skglm.utils import make_correlated_data, compiled_clone

from skglm.solvers.prox_newton import prox_newton

from statsmodels.regression import linear_model
from skglm.utils import make_correlated_data
from numpy.linalg import norm


def test_sqrt_lasso():
    n_samples, n_features = 100, 20
    rho = 0.001

    X, y, _ = make_correlated_data(n_samples, n_features, random_state=0, snr=3)

    alpha_max = norm(X.T @ y, ord=np.inf) / (np.sqrt(n_samples) * norm(y))
    alpha = rho * alpha_max

    sqrt_quad = compiled_clone(SqrtQuadratic())
    l1_penalty = compiled_clone(L1(alpha=alpha))

    w = prox_newton(X, y, sqrt_quad, l1_penalty, tol=1e-9, max_epochs=20, verbose=0)[0]

    model = linear_model.OLS(y, X)
    model = model.fit_regularized(
        method='sqrt_lasso', L1_wt=1., alpha=n_samples * alpha)
    w_stats = model.params

    np.testing.assert_almost_equal(w, w_stats, decimal=5)


# timings
if __name__ == '__main__':
    import time

    n_samples, n_features = 100, 20
    rho = 0.001
    X, y, _ = make_correlated_data(n_samples, n_features, random_state=0, snr=3)

    alpha_max = norm(X.T @ y, ord=np.inf) / (np.sqrt(n_samples) * norm(y))
    alpha = rho * alpha_max

    sqrt_quad = compiled_clone(SqrtQuadratic())
    l1_penalty = compiled_clone(L1(alpha=alpha))
    # cache numba jit compilation
    prox_newton(X, y, sqrt_quad, l1_penalty, tol=1e-9, max_epochs=20, verbose=0)[0]

    model = linear_model.OLS(y, X)
    start = time.time()
    model = model.fit_regularized(method='sqrt_lasso', L1_wt=1., alpha=n_samples*alpha)
    print("statsmodels: ", time.time() - start)
    # Output: 1.057

    start = time.time()
    prox_newton(X, y, sqrt_quad, l1_penalty, tol=1e-9, max_epochs=20, verbose=0)[0]
    print("skglm: ", time.time() - start)
    # Output: 0.077
