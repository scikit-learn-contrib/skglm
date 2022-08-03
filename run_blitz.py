import blitzl1
import numpy as np
from skglm.utils import make_correlated_data

from skglm.datafits import Logistic
from skglm.penalties import L1


n_samples, n_features = 500, 5000

X, y, _ = make_correlated_data(n_samples, n_features, random_state=0)
y = np.sign(y)

alpha_max = np.linalg.norm(X.T @ y, ord=np.inf) / 2
alpha = 0.001 * alpha_max


blitzl1.set_use_intercept(False)
blitzl1.set_tolerance(0)
blitzl1.set_verbose(True)
problem = blitzl1.LogRegProblem(X, y)

coef_ = problem.solve(alpha, max_iter=20).x

datafit = Logistic()
penalty = L1(alpha / n_samples)

obj = datafit.value(y, coef_, X @ coef_) + penalty.value(coef_)
print(n_samples * obj)