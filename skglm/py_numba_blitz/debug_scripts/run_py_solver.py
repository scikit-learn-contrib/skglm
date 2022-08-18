import numpy as np
from scipy.sparse import csc_matrix
from skglm.utils import make_correlated_data
from skglm.py_numba_blitz.solver import (py_blitz, _prox_newton_iteration,  # noqa
                                   _prox_newton_iteration_s)  # noqa
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

alpha_max = np.linalg.norm(X.T @ y, ord=np.inf) / 2
alpha = rho * alpha_max

# cache numba jit compilation
py_blitz(alpha, X, y, max_iter=10, max_epochs=10_000, tol=1e-9)

# profile code
profiler = line_profiler.LineProfiler()
profiler.add_function(py_blitz)
profiler.enable_by_count()
py_blitz(alpha, X, y, max_iter=20, max_epochs=10_000, tol=1e-9)
profiler.print_stats()
