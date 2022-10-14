import pytest

import numpy as np
from numpy.linalg import norm

from skglm.datafits import Quadratic, Logistic, QuadraticSVC
from skglm.estimators import Lasso, LinearSVC, SparseLogisticRegression
from skglm.penalties import L1, IndicatorBox
from skglm.solvers import FISTA
from skglm.utils import make_correlated_data, compiled_clone


n_samples, n_features = 10, 20
X, y, _ = make_correlated_data(
    n_samples=n_samples, n_features=n_features, random_state=0)
y_classif = np.sign(y)

alpha_max = norm(X.T @ y, ord=np.inf) / len(y)
alpha = alpha_max / 100

tol = 1e-8

# TODO: use GeneralizedLinearEstimator (to test global lipschtiz constants of every datafit)
# TODO: test sparse matrices (global lipschitz constants)
@pytest.mark.parametrize("Datafit, Penalty, Estimator", [
    (Quadratic, L1, Lasso),
    (Logistic, L1, SparseLogisticRegression),
    (QuadraticSVC, IndicatorBox, LinearSVC),
])
def test_fista_solver(Datafit, Penalty, Estimator):
    _y = y if isinstance(Datafit, Quadratic) else y_classif
    datafit = compiled_clone(Datafit())
    _init = y @ X.T if isinstance(Datafit, QuadraticSVC) else X
    datafit.initialize(_init, _y)
    penalty = compiled_clone(Penalty(alpha))

    solver = FISTA(max_iter=1000, tol=tol)
    w = solver.solve(X, _y, datafit, penalty)

    estimator = Estimator(alpha, tol=tol, fit_intercept=False)
    estimator.fit(X, _y)

    np.testing.assert_allclose(w, estimator.coef_.flatten(), rtol=1e-3)


if __name__ == '__main__':
    pass
