import numpy as np
from scipy.sparse import csc_matrix
from skglm.utils import make_correlated_data
from py_numba_blitz.solver import (py_blitz, _prox_newton_iteration,
                                   _prox_newton_iteration_s)
import line_profiler


profile = line_profiler.LineProfiler()


n_samples, n_features = 500, 5000
X_density = 1.
rho = 0.01

X, y, _ = make_correlated_data(n_samples, n_features, random_state=0,
                               X_density=X_density)
y = np.sign(y)
X = csc_matrix(X)


alpha_max = np.linalg.norm(X.T @ y, ord=np.inf) / 2
alpha = rho * alpha_max

py_blitz(alpha, X, y, max_iter=10, max_epochs=10_000,
         verbose=False, tol=1e-9, sort_ws=False)

profiler = line_profiler.LineProfiler()
profiler.add_function(_prox_newton_iteration_s)
profiler.enable_by_count()
py_blitz(alpha, X, y, max_iter=20, max_epochs=10_000,
         verbose=False, tol=1e-9, sort_ws=False)
profiler.print_stats()
