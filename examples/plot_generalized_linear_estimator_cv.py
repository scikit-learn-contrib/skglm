"""
===================================
Cross Validation for Generalized Linear Estimator
===================================
"""
import numpy as np
from sklearn.datasets import make_regression
from skglm.penalties.generalized_linear_cv import GeneralizedLinearEstimatorCV
from skglm.datafits import Quadratic
from skglm.penalties import L1_plus_L2
from skglm.solvers import AndersonCD


X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)

estimator = GeneralizedLinearEstimatorCV(
    datafit=Quadratic(),
    penalty=L1_plus_L2(alpha=1.0, l1_ratio=0.5),
    solver=AndersonCD(max_iter=50, tol=1e-4),
    cv=6,
)
estimator.fit(X, y)

# Print results
print(f"Best alpha: {estimator.alpha_:.3f}")
print(f"L1 ratio: {estimator.penalty.l1_ratio:.3f}")
print(f"Number of non-zero coefficients: {np.sum(estimator.coef_ != 0)}")

# TODO: add plot
