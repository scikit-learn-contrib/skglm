import numpy as np
from skglm.utils import make_correlated_data, compiled_clone
from skglm.penalties import L1

from skglm.prototype_PN.log_datafit import Pr_LogisticRegression
from skglm.prototype_PN.pn_solver import pn_solver


n_samples, n_features = 500, 5000

X, y, _ = make_correlated_data(n_samples, n_features, random_state=0)
y = np.sign(y)

alpha_max = np.linalg.norm(X.T @ y, ord=np.inf) / (2 * n_samples)
alpha = 0.05 * alpha_max


log_datafit = compiled_clone(Pr_LogisticRegression())
l1_penalty = compiled_clone(L1(alpha))

w = pn_solver(X, y, log_datafit, l1_penalty, tol=1e-12, verbose=1)[0]
