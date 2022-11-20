import numpy as np
from numpy.linalg import norm
import pandas as pd

from lifelines import CoxPHFitter

from skglm.estimators import GeneralizedLinearEstimator
from skglm.datafits import CoxPHBreslow
from skglm.penalties import L1
from skglm.solvers import ProxNewton
from skglm.utils import make_correlated_data


n_samples = 10
n_features = 20

X = make_correlated_data(n_samples=n_samples, n_features=n_features, density=0.3,
                         random_state=0)[0]
y = np.random.randint(0, 10, (n_samples,))
print("y:", y)

alpha_max = norm(X.T @ y, ord=np.inf) / len(y)
alpha = alpha_max * 0.1

tol = 1e-5

our_cox = GeneralizedLinearEstimator(
    datafit=CoxPHBreslow(),
    penalty=L1(alpha),
    solver=ProxNewton(max_pn_iter=100, verbose=2, fit_intercept=False, tol=tol),
).fit(X, y)

print(our_cox.coef_)
