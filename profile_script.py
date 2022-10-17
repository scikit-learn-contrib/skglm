import numpy as np
from numpy.linalg import norm
from skglm.utils import make_correlated_data, compiled_clone
from skglm.solvers import GroupProxNewton
from skglm.datafits import LogisticGroup
from skglm.penalties import WeightedGroupL2

from skglm.solvers.group_prox_newton import _descent_direction

import line_profiler


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


######
rho = 1e-1
n_groups = 100
n_samples, n_features, shuffle = 500, 5000, True
random_state = 123

X, y, _ = make_correlated_data(n_samples, n_features, rho=0.3,
                               random_state=random_state)
y = np.sign(y)

np.random.seed(random_state)
weights = np.ones(n_groups)
grp_indices, grp_ptr, _ = _generate_random_grp(n_groups, n_features, shuffle)

alpha_max = 0.
for g in range(n_groups):
    grp_g_indices = grp_indices[grp_ptr[g]: grp_ptr[g+1]]
    alpha_max = max(
        alpha_max,
        norm(X[:, grp_g_indices].T @ y) / n_samples / weights[g]
    )
alpha = rho * alpha_max


# skglm
log_group = LogisticGroup(grp_ptr=grp_ptr, grp_indices=grp_indices)
group_penalty = WeightedGroupL2(alpha, weights, grp_ptr, grp_indices)

log_group = compiled_clone(log_group, to_float32=X.dtype == np.float32)
group_penalty = compiled_clone(group_penalty)

# cache numba jit compilation
solver = GroupProxNewton(tol=1e-9, fit_intercept=False)
stop_crit = solver.solve(X, y, log_group, group_penalty)[2]
print(stop_crit)


# profile code
profiler = line_profiler.LineProfiler()
profiler.add_function(solver.solve)
profiler.enable_by_count()
solver.solve(X, y, log_group, group_penalty)
profiler.print_stats()
