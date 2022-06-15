import time
import numpy as np
from numpy.linalg import norm

from skglm.penalties.block_separable import WeightedGroupL2
from skglm.datafits.group import QuadraticGroup
from skglm.solvers.group_bcd_solver import bcd_solver

from skglm.utils import grp_converter, make_correlated_data
from celer import GroupLasso


groups, n_features = 5, 100
n_samples = 100
rnd = np.random.RandomState(0)
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
alpha = alpha_max / 10.

quad_group = QuadraticGroup(grp_ptr=grp_ptr, grp_indices=grp_indices)
group_penalty = WeightedGroupL2(
    alpha=alpha, grp_ptr=grp_ptr,
    grp_indices=grp_indices, weights=weights)
bcd_solver(X, y, quad_group, group_penalty, max_iter=1,
           tol=1e-12)[0]  # cache compilation

start = time.time()
w_skglm = bcd_solver(X, y, quad_group, group_penalty, tol=1e-12)[0]
print("time skglm:", time.time() - start)

model = GroupLasso(groups=groups, alpha=alpha, weights=weights,
                   fit_intercept=False, tol=1e-12)
start = time.time()
model.fit(X, y)
print("time celer:", time.time() - start)
w_celer = model.coef_

diff_coefs = norm(w_celer - w_skglm, ord=np.inf)
print(f"difference coefs: {diff_coefs}")

obj_skglm = quad_group.value(y, w_skglm, X @ w_skglm) + group_penalty.value(w_skglm)
obj_celer = quad_group.value(y, w_celer, X @ w_celer) + group_penalty.value(w_celer)

print(f"objective skglm: {obj_skglm}")
print(f"objective celer: {obj_celer}")
