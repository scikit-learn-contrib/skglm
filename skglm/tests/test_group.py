import numpy as np
from numpy.linalg import norm
from numpy.testing import assert_allclose

from skglm.penalties.block_separable import SparseGroupL1
from skglm.datafits.multi_task import QuadraticGroup

from skglm.penalties.separable import L1
from skglm.datafits.single_task import Quadratic

from skglm.solvers.cd_solver import cd_solver_path
from skglm.solvers.group_cd import group_solver

from celer import GroupLasso
from skglm.utils import grp_converter, make_correlated_data


def test_equivalence_SparseGroupLasso_L1():
    n_features, alpha = 1000, 1.
    grp_ptr, grp_indices = grp_converter(1, n_features)
    weights = np.array([1 for _ in range(n_features)], dtype=np.float64)

    l1_penalty = L1(alpha)

    lasso_penalty = SparseGroupL1(
        alpha=alpha, tau=1.,
        grp_ptr=grp_ptr, grp_indices=grp_indices,
        weights=weights
    )

    single_group_penalty = SparseGroupL1(
        alpha=alpha, tau=0.,
        grp_ptr=grp_ptr, grp_indices=grp_indices,
        weights=weights
    )

    penalties = [l1_penalty, lasso_penalty, single_group_penalty]

    rnd = np.random.RandomState(42)
    w = rnd.normal(loc=5, scale=4, size=n_features)
    idx = rnd.choice(n_features, size=1)[0]

    values = np.array([penalty.value(w)
                       for penalty in penalties])

    proxs = np.array([
        l1_penalty.prox_1d(w[idx], 1., 0),
        lasso_penalty.prox_1feat(+w[idx:idx+1], 1., 0)[0],
        single_group_penalty.prox_1feat(+w[idx:idx+1], 1., 0)[0]
    ])

    assert_allclose(values.std(), 0)
    assert_allclose(proxs.std(), 0)


def test_equivalence_Quadratic_datafit():
    n_samples, n_features = 100, 1000
    X, y, _ = make_correlated_data(n_samples, n_features)
    grp_ptr, grp_indices = grp_converter(1, n_features)

    quad_usual = Quadratic()
    quad_group = QuadraticGroup(grp_ptr, grp_indices)

    rnd = np.random.RandomState(42)
    w = rnd.normal(loc=5, scale=4, size=n_features)
    Xw = X @ w

    assert_allclose(quad_group.value(y, w, Xw), quad_usual.value(y, w, Xw))

    quad_usual.initialize(X, y)
    quad_group.initialize(X, y)

    assert_allclose(quad_usual.lipschitz, quad_group.lipschitz)


def test_equivalence_cd_solver():
    n_samples, n_features = 100, 1000
    X, y, _ = make_correlated_data(n_samples, n_features, random_state=42)
    alpha_max = norm(X.T@y, ord=np.inf) / n_samples

    grp_ptr, grp_indices = grp_converter(1, n_features)
    weights = np.array([1 for _ in range(n_features)], dtype=np.float64)
    alpha = alpha_max / 10.

    # group solver
    quad_group = QuadraticGroup(grp_ptr, grp_indices)
    group_penalty = SparseGroupL1(
        alpha, tau=0., grp_ptr=grp_ptr, grp_indices=grp_indices, weights=weights)

    w_group_solver = group_solver(
        X, y, quad_group, group_penalty, max_iter=1000, verbose=False, stop_tol=0.)

    # usual L1 solver
    l1_penalty = L1(alpha)
    usual_quad = Quadratic()

    _, w_l1_solver, *_ = cd_solver_path(X, y, usual_quad, l1_penalty, alphas=[alpha])

    assert_allclose(w_l1_solver.flatten(), w_group_solver, atol=1e-3, rtol=1e-3)


def test_group_lasso():
    n_samples, n_features = 100, 1000
    X, y, _ = make_correlated_data(n_samples, n_features, random_state=42)

    groups = 10  # contiguous groups of 10 elements
    n_groups = n_features // groups

    grp_ptr, grp_indices = grp_converter(groups, n_features)
    weights = abs(np.random.randn(n_groups))

    alpha_max = norm(X.T@y / np.repeat(weights, groups), ord=np.inf) / n_samples
    alpha = alpha_max / 10.

    # celer group
    model = GroupLasso(groups=groups, alpha=alpha, weights=weights,
                       fit_intercept=False, tol=1e-14)
    model.fit(X, y)

    # group solver
    quad_group = QuadraticGroup(grp_ptr, grp_indices)
    group_penalty = SparseGroupL1(
        alpha, tau=0., grp_ptr=grp_ptr, grp_indices=grp_indices, weights=weights)

    w_group_solver = group_solver(
        X, y, quad_group, group_penalty, max_iter=10000,
        verbose=False, stop_tol=1e-14, p0=n_groups//10)

    assert_allclose(w_group_solver, model.coef_, atol=1e-7, rtol=1e-4)


if __name__ == '__main__':
    test_group_lasso()
    pass
