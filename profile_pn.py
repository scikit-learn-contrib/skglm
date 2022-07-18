from tabnanny import verbose
import time
import numpy as np
from sklearn.utils import check_random_state

from celer import LogisticRegression
from skglm.solvers.prox_newton_solver import prox_newton_solver
from skglm.datafits import Logistic
from skglm.penalties import L1

n_samples, n_features = 10, 12

rng = check_random_state(0)
X = rng.normal(0, 1, (n_samples, n_features))
y = np.sign(rng.normal(0, 1, (n_samples,)))

alpha = np.linalg.norm(X.T @ y, ord=np.inf) * 0.1 / n_samples

df = Logistic()
df.initialize(X, y)

pen = L1(alpha)

# sklearn
clf = LogisticRegression(
    penalty="l1", C=1/(alpha * n_samples), fit_intercept=False,
    tol=1e-10, solver="celer-pn", p0=n_features, verbose=100, max_iter=2)
t1 = time.time()
clf.fit(X, y)
t2 = time.time()
print("celer:", t2 - t1)


# skglm
# cache compilation
# w = np.zeros(n_features)
# Xw = np.zeros(n_samples)
# prox_newton_solver(X, y, df, pen, w, Xw, tol=1e-10, max_iter=1, max_epochs=1)

w = np.zeros(n_features)
Xw = np.zeros(n_samples)
t1 = time.time()
prox_newton_solver(
    X, y, df, pen, w, Xw, tol=1e-10, p0=n_features, verbose=100,
    cst_step_size=True, max_iter=2)
t2 = time.time()
print("skglm:", t2 - t1)

# comparison
np.testing.assert_allclose(w, np.ravel(clf.coef_), rtol=1e-3)
