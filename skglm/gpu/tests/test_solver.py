import pytest

import numpy as np
from numpy.linalg import norm

from skglm.gpu.cpu_solver import CPUSolver
from skglm.gpu.cupy_solver import CupySolver
from skglm.gpu.jax_solver import JaxSolver

from skglm.gpu.utils.host_utils import eval_opt_crit


@pytest.mark.parametrize("solver", [CupySolver(),
                                    CPUSolver(),
                                    JaxSolver(use_auto_diff=False),
                                    JaxSolver(use_auto_diff=True)])
def test_solves(solver):
    random_state = 1265
    n_samples, n_features = 100, 30
    reg = 1e-2

    # generate dummy data
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_samples, n_features)
    y = rng.randn(n_samples)

    # set lambda
    lmbd_max = norm(X.T @ y, ord=np.inf)
    lmbd = reg * lmbd_max

    w = solver.solve(X, y, lmbd)

    stop_crit = eval_opt_crit(X, y, lmbd, w)
    np.testing.assert_allclose(stop_crit, 0., atol=1e-9)
