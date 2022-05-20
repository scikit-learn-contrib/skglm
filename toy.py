import numpy as np

from skglm.datafits import Quadratic
from skglm.penalties import SCAD
from skglm.solvers.cd_solver import cd_solver


n_samples = 1000
n_features = 100_000

np.random.seed(0)
X = np.random.normal(0, 1, (n_samples, n_features))
y = np.random.normal(0, 1, (n_samples,))

w = np.zeros(n_features)
Xw = np.zeros(n_samples)

alpha_max = np.linalg.norm(X.T @ y, ord=np.inf) / n_samples

df = Quadratic()
pen = SCAD(alpha_max * 0.1, 3.)
df.initialize(X, y)

cd_solver(X, y, df, pen, w, Xw, use_acc=True, verbose=2, ws_strategy="subdiff")

