import numpy as np

from skglm.datafits import Poisson
from skglm.penalties import WeightedL1
from skglm.solvers import ProxNewton
from benchopt.datasets.simulated import make_correlated_data

from skglm.utils import compiled_clone

X, y, _ = make_correlated_data(n_samples=500, n_features=400, random_state=0)
y = np.abs(y)

pen = compiled_clone(WeightedL1(1, np.zeros(X.shape[1])))
df = compiled_clone(Poisson())

df.initialize(X, y)

solver = ProxNewton(verbose=2, tol=1e-8)
w = np.zeros(X.shape[1])
Xw = np.zeros(X.shape[0])
solver.solve(X, y, df, pen, w, Xw)
