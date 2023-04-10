import pytest

import numpy as np
from numpy.linalg import norm

from skglm.gpu.solvers import CPUSolver
from skglm.gpu.solvers.base import BaseQuadratic, BaseL1

from skglm.gpu.utils.host_utils import eval_opt_crit


@pytest.mark.parametrize("solver, datafit_cls, penalty_cls",
                         [CPUSolver(), BaseQuadratic, BaseL1])
def test_solves(solver, datafit_cls, penalty_cls):
    random_state = 1265
    n_samples, n_features = 100, 30
    reg = 1e-2

    # generate dummy data
    rng = np.random.RandomState(random_state)
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
