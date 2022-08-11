import numpy as np
from skglm.utils import make_correlated_data
from py_numba_blitz.solver import py_blitz


n_samples, n_features = 100, 200
rho = 0.1
X, y, _ = make_correlated_data(n_samples, n_features, random_state=0)
y = np.sign(y)


alpha_max = np.linalg.norm(X.T @ y, ord=np.inf) / 2
alpha = rho * alpha_max

py_blitz(alpha, X, y, max_iter=5, max_epochs=10_000,
         verbose=True, tol=1e-9, sort_ws=True)
