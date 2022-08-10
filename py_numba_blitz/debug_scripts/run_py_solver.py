import numpy as np
from skglm.utils import make_correlated_data
from py_numba_blitz.solver import py_blitz


n_samples, n_features = 100, 20
rho = 0.1
X, y, _ = make_correlated_data(n_samples, n_features, random_state=0)
y = np.sign(y)


alpha_max = np.linalg.norm(X.T @ y, ord=np.inf) / 2
alpha = rho * alpha_max

py_blitz(alpha, X, y, max_iter=4, max_epochs=1000, verbose=True)


# Iter: 1 Objective: 40.62389259 Dual: 19.85152433 Duality gap: 20.77236825 Features left: 20
# Iter: 2 Objective: 40.30547455 Dual: 37.45257509 Duality gap: 2.85289946 Features left: 20
