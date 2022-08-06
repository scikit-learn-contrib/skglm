import numpy as np
from scipy.sparse import issparse
from skglm.utils import make_correlated_data, compiled_clone
from skglm.prototype_PN.L1_penalty import L1

from skglm.prototype_PN.log_datafit import Pr_LogisticRegression
from skglm.prototype_PN.pn_solver import pn_solver
from libsvmdata import fetch_libsvm


n_samples, n_features = 10, 100

X, y, _ = make_correlated_data(n_samples, n_features, random_state=0, X_density=0.1)
y = np.sign(y)

# X, y = fetch_libsvm('rcv1.binary')

alpha_max = np.linalg.norm(X.T @ y, ord=np.inf) / (2 * len(y))
alpha = 0.001 * alpha_max

log_datafit = compiled_clone(Pr_LogisticRegression())
l1_penalty = compiled_clone(L1(alpha))

w, obj_out, _ = pn_solver(X, y, log_datafit, l1_penalty,
                          tol=1e-9, use_acc=True, verbose=2)


# print(obj_out)

# obj = log_datafit.value(y, w, X @ w) + l1_penalty.value(w)
# print(n_samples * obj)
