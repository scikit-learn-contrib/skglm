# to install Blitz
# pip install git+https://github.com/QB3/BlitzL1.git@understand-blitz
import blitzl1
import numpy as np
from skglm.utils import make_correlated_data


n_samples, n_features = 100, 200
rho = 0.1
X, y, _ = make_correlated_data(n_samples, n_features, random_state=0)
y = np.sign(y)


alpha_max = np.linalg.norm(X.T @ y, ord=np.inf) / 2
alpha = rho * alpha_max


blitzl1.set_use_intercept(False)
blitzl1.set_tolerance(0)
blitzl1.set_verbose(True)
problem = blitzl1.LogRegProblem(X, y)

coef_ = problem.solve(alpha, max_iter=20).x
