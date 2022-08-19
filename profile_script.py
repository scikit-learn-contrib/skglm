import numpy as np
from scipy.sparse import csc_matrix
from skglm.utils import make_correlated_data, compiled_clone
from skglm.solvers.pn_solver_improved import pn_solver_improved
from skglm.datafits.single_task import Logistic
from skglm.penalties.separable import L1

import line_profiler


# params
n_samples, n_features = 500, 5000
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


log_datafit = compiled_clone(Logistic())
l1_penalty = compiled_clone(L1(alpha))

# cache numba jit compilation
pn_solver_improved(X, y, log_datafit, l1_penalty, max_iter=25, tol=1e-9)

# profile code
profiler = line_profiler.LineProfiler()
profiler.add_function(pn_solver_improved)
profiler.enable_by_count()
pn_solver_improved(X, y, log_datafit, l1_penalty, max_iter=25, tol=1e-9)
profiler.print_stats()
