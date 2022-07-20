import time
import blitzl1
import numpy as np
from sklearn.utils import check_random_state

from celer import LogisticRegression, celer_path
from skglm.solvers.prox_newton_solver import prox_newton_solver
from skglm.datafits import Logistic
from skglm.penalties import L1

n_samples, n_features = 100, 1_000

rng = check_random_state(0)
X = rng.normal(0, 1, (n_samples, n_features))
y = np.sign(rng.normal(0, 1, (n_samples,)))

alpha = np.linalg.norm(X.T @ y, ord=np.inf) * 0.1 / n_samples

# prob = blitzl1.LogRegProblem(X, y)
# sol = prob.solve(alpha)


df = Logistic()
df.initialize(X, y)

pen = L1(alpha)
tol = 1e-10
# max_iter = 1
# max_epochs = 1
# max_pn_cd_epochs = 2
eps_in = 1e-10


w = np.zeros(n_features)
Xw = np.zeros(n_samples)
t1 = time.time()
prox_newton_solver(
    X, y, df, pen, w, Xw, tol=tol, p0=5, verbose=100, eps_in=eps_in)
t2 = time.time()
print("skglm:", t2 - t1)
