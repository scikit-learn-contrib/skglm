import pytest

import numpy as np
from numpy.linalg import norm
from scipy.sparse import csc_matrix, issparse

from skglm.datafits import Quadratic, Logistic, QuadraticSVC
from skglm.penalties import L1, IndicatorBox
from skglm.solvers import FISTA, AndersonCD
from skglm.utils import make_correlated_data, compiled_clone


np.random.seed(0)
n_samples, n_features = 50, 60
X, y, _ = make_correlated_data(
    n_samples=n_samples, n_features=n_features, random_state=0)
X_sparse = csc_matrix(X * np.random.binomial(1, 0.1, X.shape))
y_classif = np.sign(y)

alpha_max = norm(X.T @ y, ord=np.inf) / len(y)
alpha = alpha_max / 100

tol = 1e-10


@pytest.mark.parametrize("X", [X, X_sparse])
@pytest.mark.parametrize("Datafit, Penalty", [
    (Quadratic, L1),
    (Logistic, L1),
    (QuadraticSVC, IndicatorBox),
])
def test_fista_solver(X, Datafit, Penalty):
    _y = y if isinstance(Datafit, Quadratic) else y_classif
    datafit = compiled_clone(Datafit())
    _init = y @ X.T if isinstance(Datafit, QuadraticSVC) else X
    if issparse(X):
        datafit.initialize_sparse(_init.data, _init.indptr, _init.indices, _y)
    else:
        datafit.initialize(_init, _y)
    penalty = compiled_clone(Penalty(alpha))

    solver = FISTA(max_iter=1000, tol=tol) 
    res_fista = solver.solve(X, _y, datafit, penalty)

    solver_cd = AndersonCD(tol=tol, fit_intercept=False)
    res_cd = solver_cd.solve(X, _y, datafit, penalty)

    np.testing.assert_allclose(res_fista[0], res_cd[0], rtol=1e-3)


if __name__ == '__main__':
    pass
