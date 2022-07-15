import time
import numpy as np
from skglm.solvers.prox_newton_solver import prox_newton_solver
from skglm.datafits import Logistic
from skglm.penalties import L1
from sklearn.linear_model import LogisticRegression


n_samples, n_features = 10, 100

X = np.random.normal(0, 1, (n_samples, n_features))
y = np.sign(np.random.normal(0, 1, (n_samples,)))

alpha = np.linalg.norm(X.T @ y, ord=np.inf) * 0.1 / n_samples

df = Logistic()
df.initialize(X, y)

pen = L1(alpha)

# sklearn
clf = LogisticRegression(penalty="l1", C=1/(alpha * n_samples), fit_intercept=False, 
                         tol=1e-10, solver="liblinear")
t1 = time.time()
clf.fit(X, y)
t2 = time.time()
print("sklearn:", t2 - t1)


# skglm
# cache compilation
w = np.zeros(n_features)
Xw = np.zeros(n_samples)
prox_newton_solver(X, y, df, pen, w, Xw, tol=1e-10, max_iter=1, max_epochs=1)

w = np.zeros(n_features)
Xw = np.zeros(n_samples)
t1 = time.time()
prox_newton_solver(X, y, df, pen, w, Xw, tol=1e-10)
t2 = time.time()
print("skglm:", t2 - t1)

# comparison
np.testing.assert_allclose(w, np.ravel(clf.coef_), rtol=1e-3)
