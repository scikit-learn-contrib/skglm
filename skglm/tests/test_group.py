import pytest
import numpy as np
from numpy.linalg import norm

from skglm.penalties import L1
from skglm.datafits import Quadratic
from skglm.penalties.block_separable import WeightedGroupL2
from skglm.datafits.group import QuadraticGroup
from skglm.solvers.group_bcd_solver import group_bcd_solver

from skglm.utils import (
    alpha_max_group_lasso, grp_converter, make_correlated_data, AndersonAcceleration)
from skglm.utils import compiled_clone
from celer import GroupLasso, Lasso


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


def test_check_group_compatible():
    l1_penalty = L1(1e-3)
    quad_datafit = Quadratic()
    X, y = np.random.randn(5, 5), np.random.randn(5)

    with np.testing.assert_raises(Exception):
        group_bcd_solver(X, y, quad_datafit, l1_penalty)


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

    # compile classes
    quad_group = compiled_clone(quad_group, to_float32=X.dtype == np.float32)
    group_penalty = compiled_clone(group_penalty)
    w = group_bcd_solver(X, y, quad_group, group_penalty, tol=1e-12)[0]

    np.testing.assert_allclose(norm(w), 0, atol=1e-14)


def test_equivalence_lasso():
    n_samples, n_features = 30, 50
    rnd = np.random.RandomState(1123)
    X, y, _ = make_correlated_data(n_samples, n_features, random_state=rnd)

    grp_indices, grp_ptr = grp_converter(1, n_features)
    weights = abs(rnd.randn(n_features))

    alpha_max = norm(X.T @ y / weights, ord=np.inf) / n_samples
    alpha = alpha_max / 10.

    quad_group = QuadraticGroup(grp_ptr=grp_ptr, grp_indices=grp_indices)
    group_penalty = WeightedGroupL2(
        alpha=alpha, grp_ptr=grp_ptr,
        grp_indices=grp_indices, weights=weights)

    # compile classes
    quad_group = compiled_clone(quad_group, to_float32=X.dtype == np.float32)
    group_penalty = compiled_clone(group_penalty)
    w = group_bcd_solver(X, y, quad_group, group_penalty, tol=1e-12)[0]

    celer_lasso = Lasso(
        alpha=alpha, fit_intercept=False, tol=1e-12, weights=weights).fit(X, y)

    np.testing.assert_allclose(celer_lasso.coef_, w)


@pytest.mark.parametrize("n_groups, n_features, shuffle",
                         [[15, 50, True], [5, 50, False], [19, 59, False]])
def test_vs_celer_grouplasso(n_groups, n_features, shuffle):
    n_samples = 100
    rnd = np.random.RandomState(42)
    X, y, _ = make_correlated_data(n_samples, n_features, random_state=rnd)

    grp_indices, grp_ptr, groups = _generate_random_grp(n_groups, n_features, shuffle)
    weights = abs(rnd.randn(n_groups))

    alpha_max = alpha_max_group_lasso(X, y, n_groups, grp_indices, grp_ptr, weights)
    alpha = alpha_max / 10.

    quad_group = QuadraticGroup(grp_ptr=grp_ptr, grp_indices=grp_indices)
    group_penalty = WeightedGroupL2(
        alpha=alpha, grp_ptr=grp_ptr,
        grp_indices=grp_indices, weights=weights)

    # compile classes
    quad_group = compiled_clone(quad_group, to_float32=X.dtype == np.float32)
    group_penalty = compiled_clone(group_penalty)
    w = group_bcd_solver(X, y, quad_group, group_penalty, tol=1e-12)[0]

    model = GroupLasso(groups=groups, alpha=alpha, weights=weights,
                       fit_intercept=False, tol=1e-12)
    model.fit(X, y)

    np.testing.assert_allclose(model.coef_, w, atol=1e-5)


@pytest.mark.parametrize("n_groups, n_features, shuffle",
                         [[15, 50, False]])
def test_intercept_grouplasso(n_groups, n_features, shuffle):
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

    # compile classes
    quad_group = compiled_clone(quad_group, to_float32=X.dtype == np.float32)
    group_penalty = compiled_clone(group_penalty)
    w = group_bcd_solver(
        X, y, quad_group, group_penalty, fit_intercept=True, tol=1e-12)[0]

    model = GroupLasso(groups=groups, alpha=alpha, weights=weights,
                       fit_intercept=True, tol=1e-12)
    model.fit(X, y)

    np.testing.assert_allclose(model.coef_, w[:X.shape[1]], atol=1e-5)
    np.testing.assert_allclose(model.intercept_, w[-1], atol=1e-5)


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
