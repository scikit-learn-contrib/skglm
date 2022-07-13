import pytest
import numpy as np

from numpy.linalg import norm
from numpy.testing import assert_array_less

from skglm.datafits import Quadratic
from skglm.penalties import (
    L1, L1_plus_L2, WeightedL1, MCPenalty, SCAD, IndicatorBox, L0_5, L2_3)
from skglm import GeneralizedLinearEstimator
from skglm.utils import make_correlated_data

X, y, _ = make_correlated_data(n_samples=20, n_features=10, random_state=0)

n_samples, n_features = X.shape
alpha_max = norm(X.T @ y, ord=np.inf) / n_samples
alpha = alpha_max / 1000

penalties = [
    L1(alpha=alpha),
    L1_plus_L2(alpha=alpha, l1_ratio=0.5),
    WeightedL1(alpha=1, weights=np.arange(n_features)),
    MCPenalty(alpha=alpha, gamma=4),
    SCAD(alpha=alpha, gamma=4),
    IndicatorBox(alpha=alpha),
    L0_5(alpha),
    L2_3(alpha)]


@pytest.mark.parametrize('penalty', penalties)
def test_subdiff_diff(penalty):
    estimator_ours = GeneralizedLinearEstimator(
        datafit=Quadratic(),
        penalty=L1(alpha=1),
        tol=1e-14,
    ).fit(X, y)
    # assert that something was fitted:
    assert_array_less(1e-5, norm(estimator_ours.coef_))
    # assert the stopping criterion is satisfied
    assert_array_less(estimator_ours.stop_crit_, estimator_ours.tol)


if __name__ == '__main__':
    test_subdiff_diff(L1(alpha=alpha))
    test_subdiff_diff(MCPenalty(alpha=alpha, gamma=4))
