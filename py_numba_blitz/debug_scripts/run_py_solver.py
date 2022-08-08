import numpy as np
from skglm.utils import make_correlated_data
from py_numba_blitz.solver import py_blitz


n_samples, n_features = 10, 20
rho = 0.5

X, y, _ = make_correlated_data(n_samples, n_features, random_state=0)

alpha_max = np.linalg.norm(X.T @ y, ord=np.inf) / 2
alpha = rho * alpha_max

py_blitz(alpha, X, y, max_iter=1, max_epochs=1)
