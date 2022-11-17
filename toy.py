# CoxPHFitter and lifelines as test dependencies
# Implement first Breslow method
# Then Efron's method for tied times
# Write an example for exhaustivity

import numpy as np
from numpy.linalg import norm
import pandas as pd

from lifelines import CoxPHFitter

from skglm.estimators import GeneralizedLinearEstimator
from skglm.datafits import CoxPHBreslow
from skglm.penalties import L1
from skglm.solvers import ProxNewton


df = pd.DataFrame({
    'T': [5, 3, 9, 8, 7, 4, 1, 10, 2, 11, 6, 70],
    'E': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    'var': [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2],
    'month': [10, 3, 9, 8, 7, 4, 4, 3, 2, 5, 6, 7],
    'age': [4, 3, 9, 8, 7, 4, 4, 3, 2, 5, 6, 7],
})
X = df[["var", "month", "age"]].to_numpy()
y = df["T"].to_numpy()

alpha_max = norm(X.T @ y, ord=np.inf) / len(y)
alpha = alpha_max * 0.01

tol = 1e-8

our_cox = GeneralizedLinearEstimator(
    datafit=CoxPHBreslow(),
    penalty=L1(alpha),
    solver=ProxNewton(verbose=2, fit_intercept=False, tol=tol),
).fit(X, y)

cph = CoxPHFitter(penalizer=alpha, l1_ratio=1.)
cph.fit(df, "T", "E")  # fit right censoring

np.testing.assert_allclose(our_cox.coef_, cph.params_, rtol=1e-3)
