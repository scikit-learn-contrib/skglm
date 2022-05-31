import numpy as np
from numpy.linalg import norm

from skglm.penalties.block_separable import WeightedGroupL1
from skglm.datafits.group import QuadraticGroup
from skglm.solvers.group_bcd_solver import bcd_solver

from skglm.utils import grp_converter, make_correlated_data


random_state = 1563
groups = 250
n_samples, n_features = 100, 1000
rnd = np.random.RandomState(random_state)
X, y, _ = make_correlated_data(n_samples, n_features, random_state=rnd)

grp_indices, grp_ptr = grp_converter(groups, n_features)
n_groups = len(grp_ptr) - 1
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
    alpha=alpha_max, grp_ptr=grp_ptr,
    grp_indices=grp_indices, weights=weights)

w_group_solver = bcd_solver(
    X, y, quad_group, group_penalty, max_iter=10000,
    verbose=True, tol=0)

print(norm(w_group_solver, ord=np.inf))
