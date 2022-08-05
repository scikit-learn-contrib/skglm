import numpy as np
from skglm.utils import make_correlated_data, compiled_clone
from skglm.penalties import L1

from skglm.prototype_PN.log_datafit import Pr_LogisticRegression
from skglm.prototype_PN.pn_solver import pn_solver


n_samples, n_features = 10, 5

X, y, _ = make_correlated_data(n_samples, n_features, random_state=0, X_density=0.1)
y = np.sign(y)

print(X.todense())

alpha_max = np.linalg.norm(X.T @ y, ord=np.inf) / (2 * n_samples)
alpha = 0.01 * alpha_max


log_datafit = compiled_clone(Pr_LogisticRegression())
l1_penalty = compiled_clone(L1(alpha))

w, obj_out, _ = pn_solver(X, y, log_datafit, l1_penalty,
                          tol=1e-12, use_acc=True, verbose=True)


# print(obj_out)

# obj = log_datafit.value(y, w, X @ w) + l1_penalty.value(w)
# print(n_samples * obj)
