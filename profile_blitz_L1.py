import time
import blitzl1
import numpy as np
from sklearn.utils import check_random_state

from celer import LogisticRegression, celer_path
from skglm.solvers.prox_newton_solver import prox_newton_solver
from skglm.datafits import Logistic
from skglm.penalties import L1

n_samples, n_features = 1_000, 10_000

rng = check_random_state(0)
X = rng.normal(0, 1, (n_samples, n_features))
y = np.sign(rng.normal(0, 1, (n_samples,)))

alpha_max = np.linalg.norm(X.T @ y, ord=np.inf) / 2
alpha = alpha_max / 100

tol = 1e-10

blitzl1.set_use_intercept(False)
blitzl1.set_tolerance(0)
blitzl1.set_verbose(True)

prob = blitzl1.LogRegProblem(X, y)
sol = prob.solve(alpha, p0=n_features)

# df = Logistic()
# df.initialize(X, y)

# pen = L1(alpha / n_samples)
# # max_iter = 1
# # max_epochs = 1
# # max_pn_cd_epochs = 2
# eps_in = 0.3


# w = np.zeros(n_features)
# Xw = np.zeros(n_samples)
# prox_newton_solver(
#     X, y, df, pen, w, Xw, tol=tol, p0=n_features, verbose=100, eps_in=eps_in)

# np.testing.assert_allclose(sol.x, w, rtol=1e-5)
