from tabnanny import verbose
import time
import numpy as np
from sklearn.utils import check_random_state

from celer import LogisticRegression, celer_path
from skglm.solvers.prox_newton_solver import prox_newton_solver
from skglm.datafits import Logistic
from skglm.penalties import L1

n_samples, n_features = 10, 5

rng = check_random_state(0)
X = rng.normal(0, 1, (n_samples, n_features))
y = np.sign(rng.normal(0, 1, (n_samples,)))

alpha = np.linalg.norm(X.T @ y, ord=np.inf) * 0.1 / n_samples

df = Logistic()
df.initialize(X, y)

pen = L1(alpha)
tol = 1e-10
max_iter = 1
max_epochs = 1
max_pn_cd_epochs = 10

t1 = time.time()
w_celer = celer_path(
    X, y,pb="logreg", n_alphas=1, alphas=[alpha * n_samples],
    max_iter=max_iter, max_epochs=50000, p0=n_features, verbose=100, tol=tol, use_PN=True, max_pn_iter=max_epochs, max_cd_itr=max_pn_cd_epochs)[1]
t2 = time.time()
print("celer:", t2 - t1)


w = np.zeros(n_features)
Xw = np.zeros(n_samples)
t1 = time.time()
prox_newton_solver(
    X, y, df, pen, w, Xw, tol=tol, p0=n_features, verbose=100,
    cst_step_size=True, max_iter=max_iter, max_epochs=max_epochs, max_pn_cd_epochs=max_pn_cd_epochs)
t2 = time.time()
print("skglm:", t2 - t1)

# comparison
np.testing.assert_allclose(w, np.ravel(w_celer), rtol=1e-3)
