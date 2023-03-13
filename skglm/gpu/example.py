import time

import numpy as np
from numpy.linalg import norm

from sklearn.linear_model import Lasso
from skglm.gpu.cupy_solver import CupySolver
from skglm.gpu.jax_solver import JaxSolver

from skglm.gpu.utils.host_utils import compute_obj, eval_opt_crit


random_state = 1265
n_samples, n_features = 100, 30
reg = 1e-2

# generate dummy data
rng = np.random.RandomState(random_state)
X = rng.randn(n_samples, n_features)
y = rng.randn(n_samples)


# set lambda
lmbd_max = norm(X.T @ y, ord=np.inf)
lmbd = reg * lmbd_max

solver = JaxSolver(verbose=1, use_auto_diff=False)

# cache grad
solver.max_iter = 2
solver.solve(X, y, lmbd)

# solve problem
start = time.perf_counter()
solver.max_iter = 1000
w_gpu = solver.solve(X, y, lmbd)
end = time.perf_counter()

print("gpu time: ", end - start)


start = time.perf_counter()
estimator = Lasso(alpha=lmbd / n_samples, fit_intercept=False)
estimator.fit(X, y)
end = time.perf_counter()
print("sklearn time: ", end - start)

w_sk = estimator.coef_


print(
    "Objective\n"
    f"gpu    : {compute_obj(X, y, lmbd, w_gpu):.8f}\n"
    f"sklearn: {compute_obj(X, y, lmbd, w_sk):.8f}"
)


print(
    "Optimality condition\n"
    f"gpu    : {eval_opt_crit(X, y, lmbd, w_gpu):.8f}\n"
    f"sklearn: {eval_opt_crit(X, y, lmbd, w_sk):.8f}"
)
