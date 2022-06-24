from benchopt.datasets.simulated import make_correlated_data
import numpy as np
from numpy.linalg import norm
from skglm.datafits import Quadratic
from skglm.penalties import L1
from skglm.solvers.cd_solver import cd_solver

random_state = 0
n_samples, n_features = 300, 3000
rho = 1

rng = np.random.RandomState(random_state)
X, y, _ = make_correlated_data(n_samples, n_features,
                               random_state=rng)


alpha_max = norm(X.T @ y, ord=np.inf) / len(y)
alpha = rho * alpha_max

datafit = Quadratic()
penalty = L1(alpha)


w = np.zeros(X.shape[1])
Xw = np.zeros(X.shape[0])

n_iter = 20
w = cd_solver(X, y, datafit, penalty, w=w, Xw=Xw,
              max_iter=n_iter, tol=1e-12, verbose=1)[0]
