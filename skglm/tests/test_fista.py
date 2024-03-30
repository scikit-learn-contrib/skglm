import pytest

import numpy as np
from numpy.linalg import norm

from scipy.sparse import csc_matrix, issparse

from skglm.penalties import L1, IndicatorBox
from skglm.solvers import FISTA, AndersonCD
from skglm.datafits import Quadratic, Logistic, QuadraticSVC

from skglm.utils.data import make_correlated_data
from skglm.utils.jit_compilation import compiled_clone


random_state = 113
n_samples, n_features = 50, 60

rng = np.random.RandomState(random_state)
X, y, _ = make_correlated_data(n_samples, n_features, random_state=rng)
rng.seed(random_state)
X_sparse = csc_matrix(X * np.random.binomial(1, 0.5, X.shape))
y_classif = np.sign(y)

alpha_max = norm(X.T @ y, ord=np.inf) / len(y)
alpha = alpha_max / 10

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
    w_fista = solver.solve(X, _y, datafit, penalty)[0]

    solver_cd = AndersonCD(tol=tol, fit_intercept=False)
    w_cd = solver_cd.solve(X, _y, datafit, penalty)[0]

    np.testing.assert_allclose(w_fista, w_cd, atol=1e-7)


if __name__ == '__main__':
    pass
