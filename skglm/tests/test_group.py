import numpy as np
from numpy.linalg import norm

from skglm.penalties.block_separable import WeightedGroupL1
from skglm.datafits.group_task import QuadraticGroup
from skglm.solvers.group_bcd_solver import grp_bcd_solver

from celer import GroupLasso, Lasso
from skglm.utils import grp_converter, make_correlated_data


def test_alpha_max():
    n_samples, n_features = 100, 1000
    random_state = 1563
    rnd = np.random.RandomState(random_state)
    X, y, _ = make_correlated_data(n_samples, n_features, random_state=random_state)

    groups = 20  # contiguous groups of 20 elements
    n_groups = n_features // groups

    grp_partition, grp_indices = grp_converter(groups, n_features)
    weights = abs(rnd.randn(n_groups))

    weighted_XTy = X.T@y / np.repeat(weights, groups)
    alpha_max = np.max(
        norm(weighted_XTy.reshape(-1, groups), ord=2, axis=1)) / n_samples

    # group solver
    quad_group = QuadraticGroup(grp_partition, grp_indices)
    group_penalty = WeightedGroupL1(
        alpha=alpha_max, grp_partition=grp_partition,
        grp_indices=grp_indices, weights=weights)

    w_group_solver = grp_bcd_solver(
        X, y, quad_group, group_penalty, max_iter=10000,
        verbose=False, tol=1e-14)

    np.testing.assert_equal(w_group_solver, np.zeros(n_features))


def test_equivalence_lasso():
    n_samples, n_features = 100, 1000
    random_state = 1123
    rnd = np.random.RandomState(random_state)
    X, y, _ = make_correlated_data(n_samples, n_features, random_state=random_state)

    grp_partition, grp_indices = grp_converter(1, n_features)
    weights = abs(rnd.randn(n_features))

    alpha_max = norm(X.T@y, ord=np.inf) / n_samples
    alpha = alpha_max / 10.

    # group solver
    quad_group = QuadraticGroup(grp_partition, grp_indices)
    group_penalty = WeightedGroupL1(
        alpha=alpha, grp_partition=grp_partition,
        grp_indices=grp_indices, weights=weights)

    w_group_solver = grp_bcd_solver(
        X, y, quad_group, group_penalty, max_iter=10000,
        verbose=False, tol=1e-14)

    # celer lasso
    celer_lasso = Lasso(alpha=alpha, fit_intercept=False, tol=1e-14, weights=weights)
    celer_lasso.fit(X, y)

    np.testing.assert_allclose(celer_lasso.coef_, w_group_solver, atol=1e-4, rtol=1e-3)


def test_vs_celer_GroupLasso():
    n_samples, n_features = 100, 1000
    random_state = 42
    rnd = np.random.RandomState(random_state)
    X, y, _ = make_correlated_data(n_samples, n_features, random_state=random_state)

    groups = 10  # contiguous groups of 10 elements
    n_groups = n_features // groups

    grp_partition, grp_indices = grp_converter(groups, n_features)
    weights = abs(rnd.randn(n_groups))

    weighted_XTy = X.T@y / np.repeat(weights, groups)
    alpha_max = np.max(
        norm(weighted_XTy.reshape(-1, groups), ord=2, axis=1)) / n_samples
    alpha = alpha_max / 10.

    # group solver
    quad_group = QuadraticGroup(grp_partition, grp_indices)
    group_penalty = WeightedGroupL1(
        alpha=alpha, grp_partition=grp_partition,
        grp_indices=grp_indices, weights=weights)

    w_group_solver = grp_bcd_solver(
        X, y, quad_group, group_penalty, max_iter=10000,
        verbose=False, tol=1e-14)

    # celer group
    model = GroupLasso(groups=groups, alpha=alpha, weights=weights,
                       fit_intercept=False, tol=1e-14)
    model.fit(X, y)

    np.testing.assert_allclose(model.coef_, w_group_solver, atol=1e-6, rtol=1e-6)


if __name__ == '__main__':
    pass
