import numpy as np
from scipy.sparse import csc_matrix
from skglm.utils import make_correlated_data, compiled_clone
from skglm.solvers.gram_cd import (gram_cd_solver)
from skglm.datafits.single_task import Logistic
from skglm.penalties.separable import L1

import line_profiler


# params
n_samples, n_features = 10000, 200
X_density = 1.
rho = 0.005
case_sparse = False


X, y, _ = make_correlated_data(n_samples, n_features, random_state=0,
                               X_density=X_density)
y = np.sign(y)
if case_sparse:
    X = csc_matrix(X)

alpha_max = np.linalg.norm(X.T @ y, ord=np.inf) / (2 * n_samples)
alpha = rho * alpha_max


l1_penalty = compiled_clone(L1(alpha))
# cache numba jit compilation
gram_cd_solver(X, y, l1_penalty, max_iter=1000, tol=1e-9, use_acc=False)

# profile code
profiler = line_profiler.LineProfiler()
profiler.add_function(gram_cd_solver)
profiler.enable_by_count()
gram_cd_solver(X, y, l1_penalty, max_iter=1000, tol=1e-9, use_acc=False)
profiler.print_stats()
