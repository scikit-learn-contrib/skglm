import time

import numpy as np
from numpy.linalg import norm

from skglm.gpu.solvers import NumbaSolver, CPUSolver

from skglm.gpu.utils.host_utils import compute_obj, eval_opt_crit


random_state = 1265
n_samples, n_features = 10_000, 500
reg = 1e-2

# generate dummy data
rng = np.random.RandomState(random_state)
X = rng.randn(n_samples, n_features)
y = rng.randn(n_samples)


# set lambda
lmbd_max = norm(X.T @ y, ord=np.inf)
lmbd = reg * lmbd_max

solver = NumbaSolver(verbose=0)
solver.max_iter = 10
solver.solve(X, y, lmbd)

# solve problem
start = time.perf_counter()
solver.max_iter = 1000
w_gpu = solver.solve(X, y, lmbd)
end = time.perf_counter()

print("gpu time: ", end - start)


solver_cpu = CPUSolver()
start = time.perf_counter()
w_cpu = solver_cpu.solve(X, y, lmbd)
end = time.perf_counter()
print("sklearn time: ", end - start)


print(
    "Objective\n"
    f"gpu    : {compute_obj(X, y, lmbd, w_gpu):.8f}\n"
    f"cpu    : {compute_obj(X, y, lmbd, w_cpu):.8f}"
)


print(
    "Optimality condition\n"
    f"gpu    : {eval_opt_crit(X, y, lmbd, w_gpu):.8f}\n"
    f"cpu    : {eval_opt_crit(X, y, lmbd, w_cpu):.8f}"
)
