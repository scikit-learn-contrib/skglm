import pytest
from itertools import product

import numpy as np
from numpy.linalg import norm

from skglm.penalties import L1
from skglm.datafits import Quadratic
from skglm import GeneralizedLinearEstimator
from skglm.penalties.block_separable import (
    WeightedL1GroupL2, WeightedGroupL2
)
from skglm.datafits.group import QuadraticGroup, LogisticGroup, PoissonGroup
from skglm.solvers import GroupBCD, GroupProxNewton

from skglm.utils.anderson import AndersonAcceleration
from skglm.utils.data import (make_correlated_data, grp_converter,
                              _alpha_max_group_lasso)

from celer import GroupLasso, Lasso
from sklearn.linear_model import LogisticRegression
from scipy import sparse


def _generate_random_grp(n_groups, n_features, shuffle=True):
    grp_indices = np.arange(n_features, dtype=np.int32)
    np.random.seed(0)
    if shuffle:
        np.random.shuffle(grp_indices)
    splits = np.random.choice(
        n_features, size=n_groups+1, replace=False).astype(np.int32)
    splits.sort()
    splits[0], splits[-1] = 0, n_features

    groups = [list(grp_indices[splits[i]: splits[i+1]])
              for i in range(n_groups)]

    return grp_indices, splits, groups


@pytest.mark.parametrize("solver", [GroupBCD, GroupProxNewton])
def test_check_group_compatible(solver):
    l1_penalty = L1(1e-3)
    quad_datafit = Quadratic()
    X, y = np.random.randn(5, 5), np.random.randn(5)

    with np.testing.assert_raises(Exception):
        solver().solve(X, y, quad_datafit, l1_penalty)


@pytest.mark.parametrize("n_groups, n_features, shuffle",
                         [[10, 50, True], [10, 50, False], [17, 53, False]])
def test_alpha_max(n_groups, n_features, shuffle):
    n_samples = 30
    rnd = np.random.RandomState(1563)
    X, y, _ = make_correlated_data(n_samples, n_features, random_state=rnd)

    grp_indices, grp_ptr, _ = _generate_random_grp(n_groups, n_features, shuffle)
    weights = abs(rnd.randn(n_groups))

    alpha_max = 0.
    for g in range(n_groups):
        grp_g_indices = grp_indices[grp_ptr[g]: grp_ptr[g+1]]
        alpha_max = max(
            alpha_max,
            norm(X[:, grp_g_indices].T @ y) / n_samples / weights[g]
        )

    quad_group = QuadraticGroup(grp_ptr=grp_ptr, grp_indices=grp_indices)
    group_penalty = WeightedGroupL2(
        alpha=alpha_max, grp_ptr=grp_ptr,
        grp_indices=grp_indices, weights=weights)

    w = GroupBCD(tol=1e-12).solve(X, y, quad_group, group_penalty)[0]

    np.testing.assert_allclose(norm(w), 0, atol=1e-14)


@pytest.mark.parametrize('positive', [False, True])
def test_equivalence_lasso(positive):
    n_samples, n_features = 30, 50
    rnd = np.random.RandomState(112)
    X, y, _ = make_correlated_data(n_samples, n_features, random_state=rnd)

    grp_indices, grp_ptr = grp_converter(1, n_features)
    weights = abs(rnd.randn(n_features))

    alpha_max = norm(X.T @ y / weights, ord=np.inf) / n_samples
    alpha = alpha_max / 100.

    quad_group = QuadraticGroup(grp_ptr=grp_ptr, grp_indices=grp_indices)
    group_penalty = WeightedGroupL2(
        alpha=alpha, grp_ptr=grp_ptr,
        grp_indices=grp_indices, weights=weights, positive=positive)

    w = GroupBCD(tol=1e-12).solve(X, y, quad_group, group_penalty)[0]

    celer_lasso = Lasso(
        alpha=alpha, fit_intercept=False, tol=1e-12, weights=weights,
        positive=positive).fit(X, y)

    np.testing.assert_allclose(celer_lasso.coef_, w)


@pytest.mark.parametrize("n_groups, n_features, shuffle",
                         [[15, 50, True], [5, 50, False], [19, 59, False]])
def test_vs_celer_grouplasso(n_groups, n_features, shuffle):
    n_samples = 100
    rnd = np.random.RandomState(42)
    X, y, _ = make_correlated_data(n_samples, n_features, random_state=rnd)

    grp_indices, grp_ptr, groups = _generate_random_grp(n_groups, n_features, shuffle)
    weights = abs(rnd.randn(n_groups))

    alpha_max = _alpha_max_group_lasso(X, y, grp_indices, grp_ptr, weights)
    alpha = alpha_max / 10.

    quad_group = QuadraticGroup(grp_ptr=grp_ptr, grp_indices=grp_indices)
    group_penalty = WeightedGroupL2(
        alpha=alpha, grp_ptr=grp_ptr,
        grp_indices=grp_indices, weights=weights)

    w = GroupBCD(tol=1e-12).solve(X, y, quad_group, group_penalty)[0]

    model = GroupLasso(groups=groups, alpha=alpha, weights=weights,
                       fit_intercept=False, tol=1e-12)
    model.fit(X, y)

    np.testing.assert_allclose(model.coef_, w, atol=1e-5)


def test_ws_strategy():
    n_features = 300
    X, y, _ = make_correlated_data(n_features=n_features, random_state=0)

    grp_indices, grp_ptr = grp_converter(3, n_features)
    n_groups = len(grp_ptr) - 1
    weights_g = np.ones(n_groups)
    alpha_max = _alpha_max_group_lasso(X, y, grp_indices, grp_ptr, weights_g)
    pen = WeightedGroupL2(
        alpha=alpha_max/10, weights=weights_g, grp_indices=grp_indices, grp_ptr=grp_ptr)

    solver = GroupBCD(ws_strategy="subdiff", verbose=3, fit_intercept=False, tol=1e-10)

    model = GeneralizedLinearEstimator(
        QuadraticGroup(grp_ptr, grp_indices), pen, solver=solver)

    model.fit(X, y)
    w_subdiff = model.coef_
    print("####")
    model.solver.ws_strategy = "fixpoint"
    model.fit(X, y)
    w_fixpoint = model.coef_
    # should not be the eaxct same solution:
    np.testing.assert_array_less(0, norm(w_fixpoint - w_subdiff))
    # but still should be close:
    np.testing.assert_allclose(w_fixpoint, w_subdiff, atol=1e-8)


def test_sparse_group():
    n_features = 30
    X, y, _ = make_correlated_data(n_features=n_features, random_state=0)

    grp_indices, grp_ptr = grp_converter(3, n_features)
    n_groups = len(grp_ptr) - 1

    weights_g = np.ones(n_groups, dtype=np.float64)
    weights_f = 0.5 * np.ones(n_features)
    pen = WeightedL1GroupL2(
        alpha=0.1, weights_groups=weights_g,
        weights_features=weights_f, grp_indices=grp_indices, grp_ptr=grp_ptr)

    solver = GroupBCD(ws_strategy="fixpoint", verbose=3,
                      fit_intercept=False, tol=1e-10)

    model = GeneralizedLinearEstimator(
        QuadraticGroup(grp_ptr, grp_indices), pen, solver=solver)

    model.fit(X, y)
    w = model.coef_.reshape(-1, 3)
    # some lines are 0:
    np.testing.assert_equal(w[0], 0)
    # some non zero lines have 0 entry
    np.testing.assert_array_less(0, norm(w[1]))
    # sign error -0 is not 0 without np.abs
    np.testing.assert_equal(np.abs(w[1, 0]), 0)


def test_intercept_grouplasso():
    n_groups, n_features, shuffle = 15, 50, False
    n_samples = 100
    rnd = np.random.RandomState(42)
    X, y, _ = make_correlated_data(n_samples, n_features, random_state=rnd)

    grp_indices, grp_ptr, groups = _generate_random_grp(n_groups, n_features, shuffle)
    weights = abs(rnd.randn(n_groups))

    alpha_max = 0.
    for g in range(n_groups):
        grp_g_indices = grp_indices[grp_ptr[g]: grp_ptr[g+1]]
        alpha_max = max(
            alpha_max,
            norm(X[:, grp_g_indices].T @ y) / n_samples / weights[g]
        )
    alpha = alpha_max / 10.

    quad_group = QuadraticGroup(grp_ptr=grp_ptr, grp_indices=grp_indices)
    group_penalty = WeightedGroupL2(
        alpha=alpha, grp_ptr=grp_ptr,
        grp_indices=grp_indices, weights=weights)

    w = GroupBCD(fit_intercept=True, tol=1e-12).solve(
        X, y, quad_group, group_penalty)[0]
    model = GroupLasso(groups=groups, alpha=alpha, weights=weights,
                       fit_intercept=True, tol=1e-12).fit(X, y)

    np.testing.assert_allclose(model.coef_, w[:n_features], atol=1e-5)
    np.testing.assert_allclose(model.intercept_, w[-1], atol=1e-5)


@pytest.mark.parametrize("solver, rho",
                         product([GroupBCD, GroupProxNewton], [1e-1, 1e-2]))
def test_equivalence_logreg(solver, rho):
    n_samples, n_features = 30, 50
    rng = np.random.RandomState(1123)
    X, y, _ = make_correlated_data(n_samples, n_features, random_state=rng)
    y = np.sign(y)

    grp_indices, grp_ptr = grp_converter(1, n_features)
    weights = np.ones(n_features)
    alpha_max = norm(X.T @ y, ord=np.inf) / (2 * n_samples)
    alpha = rho * alpha_max / 10.

    group_logistic = LogisticGroup(grp_ptr=grp_ptr, grp_indices=grp_indices)
    group_penalty = WeightedGroupL2(
        alpha=alpha, grp_ptr=grp_ptr,
        grp_indices=grp_indices, weights=weights)

    w = solver(tol=1e-12).solve(X, y, group_logistic, group_penalty)[0]

    sk_logreg = LogisticRegression(penalty='l1', C=1/(n_samples * alpha),
                                   fit_intercept=False, tol=1e-12, solver='liblinear')
    sk_logreg.fit(X, y)

    np.testing.assert_allclose(sk_logreg.coef_.flatten(), w, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("solver, n_groups, rho, fit_intercept",
                         product([GroupBCD, GroupProxNewton], [15, 25], [1e-1, 1e-2],
                                 [False, True]))
def test_group_logreg(solver, n_groups, rho, fit_intercept):
    n_samples, n_features, shuffle = 30, 60, True
    random_state = 123
    rng = np.random.RandomState(random_state)

    X, y, _ = make_correlated_data(n_samples, n_features, random_state=rng)
    y = np.sign(y)

    rng.seed(random_state)
    weights = np.abs(rng.randn(n_groups))
    grp_indices, grp_ptr, _ = _generate_random_grp(n_groups, n_features, shuffle)

    alpha_max = _alpha_max_group_lasso(X, y, grp_indices, grp_ptr, weights)
    alpha = rho * alpha_max

    # skglm
    group_logistic = LogisticGroup(grp_ptr=grp_ptr, grp_indices=grp_indices)
    group_penalty = WeightedGroupL2(alpha, weights, grp_ptr, grp_indices)

    stop_crit = solver(tol=1e-12, fit_intercept=fit_intercept).solve(
        X, y, group_logistic, group_penalty)[2]

    np.testing.assert_array_less(stop_crit, 1e-12)


def test_anderson_acceleration():
    # VAR: w = rho * w + 1 with |rho| < 1
    # converges to w_star = 1 / (1 - rho)
    max_iter, tol = 1000, 1e-9
    n_features = 2
    rho = np.array([0.5, 0.8])
    w_star = 1 / (1 - rho)
    X = np.diag([2, 5])

    # with acceleration
    acc = AndersonAcceleration(K=5)
    n_iter_acc = 0
    w = np.ones(n_features)
    Xw = X @ w
    for i in range(max_iter):
        w, Xw, _ = acc.extrapolate(w, Xw)
        w = rho * w + 1
        Xw = X @ w

        if norm(w - w_star, ord=np.inf) < tol:
            n_iter_acc = i
            break

    # without acceleration
    n_iter = 0
    w = np.ones(n_features)
    for i in range(max_iter):
        w = rho * w + 1

        if norm(w - w_star, ord=np.inf) < tol:
            n_iter = i
            break

    np.testing.assert_allclose(w, w_star)
    np.testing.assert_allclose(Xw, X @ w_star)

    np.testing.assert_array_equal(n_iter_acc, 13)
    np.testing.assert_array_equal(n_iter, 99)


@pytest.mark.parametrize("test_type", ["dense_vs_expected", "sparse_finite",
                                       "sparse_vs_dense"])
def test_poisson_group_gradient(test_type):
    np.random.seed(0)
    X = np.random.randn(15, 6)
    X[X < 0] = 0
    X_sparse = sparse.csc_matrix(X)
    y = np.random.poisson(1.0, 15)
    w = np.random.randn(6) * 0.1
    Xw = X @ w

    grp_indices, grp_ptr = grp_converter(2, 6)
    poisson_group = PoissonGroup(grp_ptr=grp_ptr, grp_indices=grp_indices)

    for g in range(2):
        if test_type == "dense_vs_expected":
            # Test dense gradient against expected
            raw_grad = poisson_group.raw_grad(y, Xw)
            group_idx = grp_indices[grp_ptr[g]:grp_ptr[g+1]]
            expected = X[:, group_idx].T @ raw_grad
            grad = poisson_group.gradient_g(X, y, w, Xw, g)
            np.testing.assert_allclose(grad, expected, rtol=1e-10)

        elif test_type == "sparse_finite":
            # Test sparse gradient is finite and has correct shape
            grad = poisson_group.gradient_g_sparse(
                X_sparse.data, X_sparse.indptr, X_sparse.indices, y, w, Xw, g
            )
            assert np.all(np.isfinite(grad))
            assert grad.shape[0] == grp_ptr[g+1] - grp_ptr[g]

        elif test_type == "sparse_vs_dense":
            # Test sparse matches dense
            grad_dense = poisson_group.gradient_g(X, y, w, Xw, g)
            grad_sparse = poisson_group.gradient_g_sparse(
                X_sparse.data, X_sparse.indptr, X_sparse.indices, y, w, Xw, g
            )
            np.testing.assert_allclose(grad_sparse, grad_dense, rtol=1e-8)


@pytest.mark.parametrize("fit_intercept,alpha", [(False, 0.1), (True, 0.01)])
def test_poisson_group_solver(fit_intercept, alpha):
    """Test solver convergence, solution quality, and intercept handling."""
    np.random.seed(0)
    X = np.random.randn(30, 9)
    true_intercept = 2.0 if fit_intercept else 0.0
    y = np.random.poisson(np.exp(0.1 * X.sum(axis=1) + true_intercept))

    grp_indices, grp_ptr = grp_converter(3, 9)
    datafit = PoissonGroup(grp_ptr=grp_ptr, grp_indices=grp_indices)
    weights = np.array([1.0, 0.5, 2.0])
    penalty = WeightedGroupL2(alpha=alpha, grp_ptr=grp_ptr,
                              grp_indices=grp_indices, weights=weights)

    w, _, stop_crit = GroupProxNewton(fit_intercept=fit_intercept, tol=1e-8).solve(
        X, y, datafit, penalty)

    assert stop_crit < 1e-8 and np.all(np.isfinite(w))

    n_features = 9
    w_coef = w[:n_features]
    intercept = w[-1] if fit_intercept else 0.0
    def loss(w, b): return np.mean(np.exp(X @ w + b) - y * (X @ w + b))
    assert loss(w_coef, intercept) < loss(np.zeros(n_features), 0.0)

    if fit_intercept:
        assert 1.0 < w[-1] < 3.0


if __name__ == "__main__":
    pass
