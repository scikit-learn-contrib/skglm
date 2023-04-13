import pytest

from scipy import sparse

import numpy as np
from numpy.linalg import norm

from skglm.gpu.solvers import CPUSolver
from skglm.gpu.solvers.base import BaseQuadratic, BaseL1

from skglm.gpu.solvers.cupy_solver import CupySolver, L1CuPy
from skglm.gpu.solvers.jax_solver import JaxSolver, QuadraticJax, L1Jax
from skglm.gpu.solvers.numba_solver import NumbaSolver, QuadraticNumba, L1Numba


from skglm.gpu.utils.host_utils import eval_opt_crit


@pytest.mark.parametrize("solver, datafit_cls, penalty_cls",
                         [(CPUSolver(), BaseQuadratic, BaseL1),
                          (CupySolver(), BaseQuadratic, L1CuPy),
                          (JaxSolver(use_auto_diff=True), QuadraticJax, L1Jax),
                          (JaxSolver(use_auto_diff=False), QuadraticJax, L1Jax),
                          (NumbaSolver(), QuadraticNumba, L1Numba)])
@pytest.mark.parametrize("sparse_X", [True, False])
def test_solves(sparse_X, solver, datafit_cls, penalty_cls):
    if sparse_X and isinstance(solver, NumbaSolver):
        pytest.xfail(reason="Sparse X is not yet supported for Numba")

    random_state = 1265
    n_samples, n_features = 100, 30
    reg = 1e-2

    # generate dummy data
    rng = np.random.RandomState(random_state)
    if sparse_X:
        X = sparse.rand(n_samples, n_features, density=0.1,
                        format="csc", random_state=rng)
    else:
        X = rng.randn(n_samples, n_features)
    y = rng.randn(n_samples)

    # set lambda
    lmbd_max = norm(X.T @ y, ord=np.inf) / n_samples
    lmbd = reg * lmbd_max

    w = solver.solve(X, y, datafit_cls(), penalty_cls(lmbd))

    stop_crit = eval_opt_crit(X, y, lmbd, w)

    np.testing.assert_allclose(stop_crit, 0., atol=1e-9)


if __name__ == "__main__":
    pass
