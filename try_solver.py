import numpy as np

from skglm.penalties import L1
from skglm.datafits import Logistic
from skglm.solvers.prox_newton import ProxNewton
from skglm.utils import compiled_clone, make_correlated_data

X, y, _ = make_correlated_data(100, 200, random_state=0)
y = np.sign(y)
pen = compiled_clone(L1(alpha=np.linalg.norm(X.T @ y, ord=np.inf) / (4 * len(y))))
df = compiled_clone(Logistic())
solver = ProxNewton(verbose=2)

solver.solve(X, y, df, pen)
