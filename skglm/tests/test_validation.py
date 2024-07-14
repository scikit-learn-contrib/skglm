import pytest

from skglm.penalties import L1
from skglm.datafits import Poisson, Huber
from skglm.solvers import FISTA, ProxNewton
from skglm.utils.data import make_correlated_data
from skglm.utils.jit_compilation import compiled_clone


def test_datafit_penalty_solver_compatibility():
    X, y, _ = make_correlated_data(n_samples=3, n_features=5)

    with pytest.raises(
        AttributeError, match="Missing `raw_grad` and `raw_hessian`"
    ):
        ProxNewton()._validate(
            X, y, compiled_clone(Huber(1.)), compiled_clone(L1(1.))
        )

    with pytest.raises(
        AttributeError, match="Missing `get_global_lipschitz`"
    ):
        FISTA()._validate(
            X, y, compiled_clone(Poisson()), compiled_clone(L1(1.))
        )

    with pytest.raises(
        AttributeError, match="Missing `get_global_lipschitz`"
    ):
        FISTA()._validate(
            X, y, compiled_clone(Poisson()), compiled_clone(L1(1.))
        )


if __name__ == "__main__":
    pass
