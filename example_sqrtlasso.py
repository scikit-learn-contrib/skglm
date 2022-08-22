import numpy as np
from numpy.linalg import norm
from skglm.penalties import L1
from skglm.datafits.single_task import SqrtQuadratic
from skglm.utils import make_correlated_data, compiled_clone

from skglm.solvers.prox_newton import prox_newton

from statsmodels.regression import linear_model
from skglm.utils import make_correlated_data
from numpy.linalg import norm


n_samples, n_features = 10, 20
rho = 0.1
X, y, _ = make_correlated_data(n_samples, n_features, random_state=0, snr=1)

alpha_max = norm(X.T @ y, ord=np.inf) / (np.sqrt(n_samples) * norm(y))
alpha = rho * alpha_max

sqrt_quad = compiled_clone(SqrtQuadratic())
l1_penalty = compiled_clone(L1(alpha=alpha))

w = prox_newton(X, y, sqrt_quad, l1_penalty, tol=1e-9, verbose=1)[0]


model = linear_model.OLS(y, X)
model = model.fit_regularized(method='sqrt_lasso', L1_wt=1., alpha=n_samples * alpha)

w_stats = model.params

print(norm(w - w_stats, ord=np.inf))
