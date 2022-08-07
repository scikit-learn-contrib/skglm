import blitzl1
import numpy as np
from skglm.utils import make_correlated_data


n_samples, n_features = 10, 20
rho = 1.

X, y, _ = make_correlated_data(n_samples, n_features)

alpha_max = np.linalg.norm(X.T @ y, ord=np.inf) / 2
alpha = rho * alpha_max


blitzl1.set_use_intercept(False)
blitzl1.set_tolerance(0)
blitzl1.set_verbose(True)
problem = blitzl1.LogRegProblem(X, y)

coef_ = problem.solve(alpha, max_iter=20).x
