import pytest
import numpy as np

from numpy.linalg import norm
from numpy.testing import assert_array_less

from sklearn.linear_model import LinearRegression

from skglm.datafits import Quadratic, QuadraticMultiTask
from skglm.penalties import (
    L1, L1_plus_L2, WeightedL1, MCPenalty, SCAD, IndicatorBox, L0_5, L2_3, SLOPE,
    LogSumPenalty, PositiveConstraint, L2_1, L2_05, BlockMCPenalty, BlockSCAD)
from skglm import GeneralizedLinearEstimator, Lasso
from skglm.solvers import AndersonCD, MultiTaskBCD, FISTA
from skglm.utils.data import make_correlated_data

from skglm.utils.prox_funcs import prox_log_sum, _log_sum_prox_val


n_samples = 20
n_features = 10
n_tasks = 10
X, Y, _ = make_correlated_data(
    n_samples=n_samples, n_features=n_features, n_tasks=n_tasks, density=0.5,
    random_state=0)
y = Y[:, 0]

n_samples, n_features = X.shape
alpha_max = norm(X.T @ y, ord=np.inf) / n_samples
alpha = alpha_max / 1000

tol = 1e-10

penalties = [
    L1(alpha=alpha),
    L1_plus_L2(alpha=alpha, l1_ratio=0.5),
    WeightedL1(alpha=1, weights=np.arange(n_features)),
    MCPenalty(alpha=alpha, gamma=4),
    SCAD(alpha=alpha, gamma=4),
    IndicatorBox(alpha=alpha),
    L0_5(alpha),
    L2_3(alpha),
    LogSumPenalty(alpha=alpha, eps=1e-2)
]

block_penalties = [
    L2_1(alpha=alpha), L2_05(alpha=alpha),
    BlockMCPenalty(alpha=alpha, gamma=4),
    BlockSCAD(alpha=alpha, gamma=4)
]


@pytest.mark.parametrize('penalty', penalties)
def test_subdiff_diff(penalty):
    # tol=1e-14 is too low when coefs are of order 1. square roots are computed in
    # some penalties and precision is lost
    est = GeneralizedLinearEstimator(
        datafit=Quadratic(),
        penalty=penalty,
        solver=AndersonCD(tol=tol)
    ).fit(X, y)
    # assert the stopping criterion is satisfied
    assert_array_less(est.stop_crit_, tol)


@pytest.mark.parametrize('block_penalty', block_penalties)
def test_subdiff_diff_block(block_penalty):
    est = GeneralizedLinearEstimator(
        datafit=QuadraticMultiTask(),
        penalty=block_penalty,
        solver=MultiTaskBCD(tol=tol)
    ).fit(X, Y)
    # assert the stopping criterion is satisfied
    assert_array_less(est.stop_crit_, est.solver.tol)


def test_slope_lasso():
    # check that when alphas = [alpha, ..., alpha], SLOPE and L1 solutions are equal
    alphas = np.full(n_features, alpha)
    est = GeneralizedLinearEstimator(
        penalty=SLOPE(alphas),
        solver=FISTA(max_iter=1000, tol=tol, opt_strategy="fixpoint"),
    ).fit(X, y)
    lasso = Lasso(alpha, fit_intercept=False, tol=tol).fit(X, y)
    np.testing.assert_allclose(est.coef_, lasso.coef_, rtol=1e-5)


def test_slope():
    # compare solutions with `sortedl1`: https://github.com/jolars/sortedl1
    try:
        from sortedl1 import Slope as SlopeEst  # noqa
    except ImportError:
        pytest.xfail(
            "This test requires slope to run.\n"
            "https://github.com/jolars/sortedl1")

    # q = 0.1
    # alphas = lambda_sequence(
    #     X, y, fit_intercept=False, reg=alpha / alpha_max, q=q)
    clf = SlopeEst(
        alpha=0.01,
        fit_intercept=False,
        scaling = "none",
        centering = "none"
    ).fit(X, y)
    alphas = clf.lambda_
    ours = GeneralizedLinearEstimator(
        penalty=SLOPE(clf.alpha * alphas),
        solver=FISTA(max_iter=1000, tol=tol, opt_strategy="fixpoint"),
    ).fit(X, y)
    np.testing.assert_allclose(ours.coef_, clf.coef_, rtol=1e-5)


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_nnls(fit_intercept):
    # compare solutions with sklearn's LinearRegression, note that n_samples >=
    # n_features for the design matrix to be injective, hence the solution unique
    clf = GeneralizedLinearEstimator(
        datafit=Quadratic(),
        penalty=PositiveConstraint(),
        solver=AndersonCD(tol=tol, fit_intercept=fit_intercept),
    ).fit(X, y)
    reg_nnls = LinearRegression(positive=True, fit_intercept=fit_intercept).fit(X, y)

    np.testing.assert_allclose(clf.coef_, reg_nnls.coef_)
    np.testing.assert_allclose(clf.intercept_, reg_nnls.intercept_)


def test_logsum_prox():
    alpha = 1.

    grid_z = np.linspace(-2, 2, num=10)
    grid_test = np.linspace(-5, 5, num=100)
    grid_eps = np.linspace(0, 5, num=10 + 1)[1:]

    for z, eps in zip(grid_z, grid_eps):
        prox = prox_log_sum(z, alpha, eps)
        obj_at_prox = _log_sum_prox_val(prox, z, alpha, eps)

        is_lowest = all(
            obj_at_prox <= _log_sum_prox_val(x, z, alpha, eps) for x in grid_test
        )

        np.testing.assert_equal(is_lowest, True)


if __name__ == "__main__":
    pass
