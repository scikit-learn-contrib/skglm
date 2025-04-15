import pytest
import numpy as np
from scipy import sparse

from skglm.penalties import L1, WeightedL1GroupL2, WeightedGroupL2
from skglm.datafits import Poisson, Huber, QuadraticGroup, LogisticGroup
from skglm.solvers import FISTA, ProxNewton, GroupBCD, GramCD, GroupProxNewton

from skglm.utils.data import grp_converter
from skglm.utils.data import make_correlated_data


def test_datafit_penalty_solver_compatibility():
    grp_size, n_features = 3, 9
    n_samples = 10
    X, y, _ = make_correlated_data(n_samples, n_features)
    X_sparse = sparse.csc_array(X)

    n_groups = n_features // grp_size
    weights_groups = np.ones(n_groups)
    weights_features = np.ones(n_features)
    grp_indices, grp_ptr = grp_converter(grp_size, n_features)

    # basic compatibility checks
    with pytest.raises(
        AttributeError, match="Missing `raw_grad` and `raw_hessian`"
    ):
        ProxNewton()._validate(
            X, y, Huber(1.), L1(1.)
        )
    with pytest.raises(
        AttributeError, match="Missing `get_global_lipschitz`"
    ):
        FISTA()._validate(
            X, y, Poisson(), L1(1.)
        )
    with pytest.raises(
        AttributeError, match="Missing `get_global_lipschitz`"
    ):
        FISTA()._validate(
            X, y, Poisson(), L1(1.)
        )
    # check Gram Solver
    with pytest.raises(
        AttributeError, match="`GramCD` supports only `Quadratic` datafit"
    ):
        GramCD()._validate(
            X, y, Poisson(), L1(1.)
        )
    # check working set strategy subdiff
    with pytest.raises(
        AttributeError, match="Penalty must implement `subdiff_distance`"
    ):
        GroupBCD()._validate(
            X, y,
            datafit=QuadraticGroup(grp_ptr, grp_indices),
            penalty=WeightedL1GroupL2(
                1., weights_groups, weights_features, grp_ptr, grp_indices)
        )
    # checks for sparsity
    with pytest.raises(
        ValueError,
        match="Sparse matrices are not yet supported in `GroupProxNewton` solver."
    ):
        GroupProxNewton()._validate(
            X_sparse, y,
            datafit=QuadraticGroup(grp_ptr, grp_indices),
            penalty=WeightedL1GroupL2(
                1., weights_groups, weights_features, grp_ptr, grp_indices)
        )
    with pytest.raises(
        AttributeError,
        match="LogisticGroup is not compatible with solver GroupBCD with sparse data."
    ):
        GroupBCD()._validate(
            X_sparse, y,
            datafit=LogisticGroup(grp_ptr, grp_indices),
            penalty=WeightedGroupL2(1., weights_groups, grp_ptr, grp_indices)
        )


if __name__ == "__main__":
    pass
