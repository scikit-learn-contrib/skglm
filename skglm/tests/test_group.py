import pytest
import numpy as np
from numpy.linalg import norm

from skglm.penalties.block_separable import WeightedGroupL2
from skglm.datafits.group import QuadraticGroup
from skglm.solvers.group_bcd_solver import bcd_solver

from skglm.utils import grp_converter, make_correlated_data
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

    w = bcd_solver(
        X, y, quad_group, group_penalty, max_iter=10000, tol=0)[0]

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

    w = bcd_solver(X, y, quad_group, group_penalty, max_iter=10000, tol=1e-12)[0]

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

    w = bcd_solver(X, y, quad_group, group_penalty, tol=1e-12)[0]

    model = GroupLasso(groups=groups, alpha=alpha, weights=weights,
                       fit_intercept=False, tol=1e-12)
    model.fit(X, y)

    np.testing.assert_allclose(model.coef_, w, atol=1e-5)


def test_gsupp():
    n_groups, n_features, shuffle = 5, 50, False
    grp_indices, grp_ptr, _ = _generate_random_grp(n_groups, n_features, shuffle)

    grp_penalty = WeightedGroupL2(
        alpha=1., grp_ptr=grp_ptr,
        grp_indices=grp_indices, weights=np.ones(n_groups))

    assert np.all(grp_penalty.is_penalized(n_groups))

    w = np.zeros(n_features)
    in_gsupp_grps = np.random.choice(n_groups, size=2, replace=False)
    out_gsupp_grps = np.setdiff1d(np.arange(n_groups), in_gsupp_grps)
    for g in in_gsupp_grps:
        grp_g_indices = grp_indices[grp_ptr[g]: grp_ptr[g+1]]
        w[grp_g_indices] = 1.

    gsupp = grp_penalty.generalized_support(w)
    assert np.all(gsupp[in_gsupp_grps])
    assert np.any(gsupp[out_gsupp_grps]) == False


if __name__ == '__main__':
    test_gsupp()
    pass
