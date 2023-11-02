import pytest
from skglm.datafits import Quadratic, Poisson
from skglm.penalties import L1
from skglm.solvers import FISTA, ProxNewton
from skglm.utils.jit_compilation import compiled_clone


def test_datafit_penalty_solver_compatibility():
    with pytest.raises(
        AttributeError, match="Missing `raw_grad` and `raw_hessian`"
    ):
        ProxNewton().validate(
            compiled_clone(Quadratic()), compiled_clone(L1(1.))
        )

    with pytest.raises(
        AttributeError, match="Missing `get_global_lipschitz`"
    ):
        FISTA().validate(
            compiled_clone(Poisson()), compiled_clone(L1(1.))
        )


if __name__ == "__main__":
    pass
