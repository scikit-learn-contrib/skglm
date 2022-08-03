import numpy as np
from skglm.utils import make_correlated_data, compiled_clone
from skglm.penalties import L1
from skglm.datafits import Logistic

from skglm.prototype_PN.pn_PAB import prox_newton_solver


n_samples, n_features = 500, 5000

X, y, _ = make_correlated_data(n_samples, n_features, random_state=0)
y = np.sign(y)

alpha_max = np.linalg.norm(X.T @ y, ord=np.inf) / (2 * n_samples)
alpha = 0.001 * alpha_max


datafit = Logistic()
datafit.initialize(X, y)
pen = L1(alpha=alpha)

pen = compiled_clone(pen)
datafit = compiled_clone(datafit)


w = np.zeros(n_features)
Xw = np.zeros(n_samples)
w_newton, obj_out, _ = prox_newton_solver(X, y, datafit, pen, w, Xw, tol=1e-12)


print(n_samples * obj_out)

obj = datafit.value(y, w, X @ w) + pen.value(w)
print(n_samples * obj)