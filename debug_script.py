import numpy as np
from numpy.linalg import norm

from skglm.penalties.block_separable import WeightedGroupL1
from skglm.datafits.group import QuadraticGroup
from skglm.solvers.group_bcd_solver import bcd_solver

from skglm.utils import grp_converter, make_correlated_data


def _generate_random_grp(n_groups, n_features, random_state=123654):
    rnd = np.random.RandomState(random_state)

    all_features = np.arange(n_features)
    rnd.shuffle(all_features)
    splits = rnd.choice(all_features, size=n_groups+1, replace=False)
    splits.sort()
    splits[0], splits[-1] = 0, n_features

    return [list(all_features[splits[i]: splits[i+1]])
            for i in range(n_groups)]


random_state = 1563
n_samples, n_features, n_groups = 100, 1000, 60
groups = _generate_random_grp(n_groups, n_features)

rnd = np.random.RandomState(random_state)
X, y, _ = make_correlated_data(n_samples, n_features, random_state=rnd)

grp_indices, grp_ptr = grp_converter(groups, n_features)
weights = abs(rnd.randn(n_groups))

alpha_max = 0.
for g in range(n_groups):
    grp_g_indices = grp_indices[grp_ptr[g]: grp_ptr[g+1]]
    alpha_max = max(
        alpha_max,
        norm(X[:, grp_g_indices].T @ y) / n_samples / weights[g]
    )

# group solver
quad_group = QuadraticGroup(grp_ptr=grp_ptr, grp_indices=grp_indices)
group_penalty = WeightedGroupL1(
    alpha=alpha_max * 1.0000001, grp_ptr=grp_ptr,
    grp_indices=grp_indices, weights=weights)

w_group_solver = bcd_solver(
    X, y, quad_group, group_penalty, max_iter=10000,
    verbose=True, tol=1e-10)

print(norm(w_group_solver, ord=np.inf))
