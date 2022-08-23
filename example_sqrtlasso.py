import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from statsmodels.regression import linear_model

from skglm.penalties import L1
from skglm.datafits.single_task import SqrtQuadratic
from skglm.utils import make_correlated_data, compiled_clone
from skglm.solvers.prox_newton import prox_newton
from skglm.utils import make_correlated_data


def test_alpha_max():
    n_samples, n_features = 50, 10
    X, y, _ = make_correlated_data(n_samples, n_features, random_state=0)

    alpha_max = norm(X.T @ y, ord=np.inf) / (np.sqrt(n_samples) * norm(y))

    sqrt_quad = compiled_clone(SqrtQuadratic())
    l1_penalty = compiled_clone(L1(alpha=alpha_max))
    w = prox_newton(X, y, sqrt_quad, l1_penalty, tol=1e-9)[0]

    np.testing.assert_equal(w, 0)


def test_vs_statmodels():
    n_samples, n_features = 100, 20
    rho = 0.001

    X, y, _ = make_correlated_data(n_samples, n_features, random_state=0)

    alpha_max = norm(X.T @ y, ord=np.inf) / (np.sqrt(n_samples) * norm(y))
    alpha = rho * alpha_max

    sqrt_quad = compiled_clone(SqrtQuadratic())
    l1_penalty = compiled_clone(L1(alpha=alpha))

    w = prox_newton(X, y, sqrt_quad, l1_penalty, tol=1e-9, max_epochs=20)[0]

    model = linear_model.OLS(y, X)
    model = model.fit_regularized(
        method='sqrt_lasso', L1_wt=1., alpha=n_samples * alpha)
    w_stats = model.params

    np.testing.assert_almost_equal(w, w_stats, decimal=5)


# timings
if __name__ == '__main__':
    import time

    n_samples, n_features = 300, 500
    X, y, _ = make_correlated_data(n_samples, n_features, random_state=0, snr=3)

    alpha_max = norm(X.T @ y, ord=np.inf) / (np.sqrt(n_samples) * norm(y))

    rhos = np.geomspace(0.05, 1, 50)
    times_stat = []
    times_skglm = []
    objs_stat = []
    objs_skglm = []
    norm_res = []
    supp_size = []

    for idx, rho in enumerate(rhos):
        alpha = rho * alpha_max

        sqrt_quad = compiled_clone(SqrtQuadratic())
        l1_penalty = compiled_clone(L1(alpha=alpha))

        model = linear_model.OLS(y, X)
        start = time.time()
        model = model.fit_regularized(
            method='sqrt_lasso', L1_wt=1., alpha=n_samples*alpha)
        times_stat.append(time.time() - start)

        start = time.time()
        sol = prox_newton(X, y, sqrt_quad, l1_penalty,
                          tol=1e-9, max_epochs=20, verbose=1)[0]
        times_skglm.append(time.time() - start)

        supp_size.append((sol != 0).sum())
        objs_stat.append(norm(y - X @ model.params) /
                         np.sqrt(n_samples) + alpha * norm(model.params, 1))
        objs_skglm.append(norm(y - X @ sol) / np.sqrt(n_samples) + alpha * norm(sol, 1))

        norm_res.append(norm(y - X @ sol))

plt.close('all')
fig, axarr = plt.subplots(4, 1, sharex=True, constrained_layout=True)
axarr[0].semilogy(rhos, norm_res)
axarr[0].set_ylabel("residual norm")
axarr[1].plot(rhos, supp_size)
axarr[1].set_ylabel("support size")
axarr[2].semilogy(rhos, np.array(objs_skglm) - np.array(objs_stat))
axarr[2].set_ylabel("obj skglm - obj stat")
axarr[3].set_xlabel(r'$\lambda/\lambda_{\max}$')
axarr[3].semilogy(rhos, np.array(times_skglm) / np.array(times_stat))
axarr[3].set_ylabel("$t_{skglm} / t_{statmodels}$")
plt.show(block=False)
