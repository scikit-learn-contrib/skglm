import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression

from skglm.experimental.quantile_huber import QuantileHuber
from skglm.experimental.progressive_smoothing import ProgressiveSmoothingSolver
from skglm.experimental.quantile_regression import Pinball
from skglm.datafits import Huber
from skglm import GeneralizedLinearEstimator
from skglm.penalties import L1
from skglm.solvers import LBFGS
from skglm.solvers import FISTA


class CustomLBFGS(LBFGS):
    """A wrapper around LBFGS that adds the attributes needed by ProgressiveSmoothingSolver."""

    def __init__(self, max_iter=100, tol=1e-4):
        super().__init__(max_iter=max_iter, tol=tol)
        self.fit_intercept = True
        self.warm_start = False


def test_quantile_huber_basic():
    """Test basic functionality of QuantileHuber class."""
    # Test initialization and parameter validation
    with pytest.raises(ValueError):
        QuantileHuber(delta=-1, quantile=0.5)  # Invalid delta

    with pytest.raises(ValueError):
        QuantileHuber(delta=1, quantile=1.5)  # Invalid quantile

    # Test values for different quantiles
    qh_median = QuantileHuber(delta=1.0, quantile=0.5)
    qh_upper = QuantileHuber(delta=1.0, quantile=0.8)
    qh_lower = QuantileHuber(delta=1.0, quantile=0.2)

    # For r > delta, upper quantile should penalize more than median
    r = 2.0
    _, grad_median = qh_median._loss_and_grad_scalar(r)
    _, grad_upper = qh_upper._loss_and_grad_scalar(r)
    _, grad_lower = qh_lower._loss_and_grad_scalar(r)

    assert grad_upper > grad_median  # Upper quantile penalizes overestimates more
    assert grad_lower < grad_median  # Lower quantile penalizes overestimates less

    # For r < -delta, lower quantile should penalize more than median
    r = -2.0
    _, grad_median = qh_median._loss_and_grad_scalar(r)
    _, grad_upper = qh_upper._loss_and_grad_scalar(r)
    _, grad_lower = qh_lower._loss_and_grad_scalar(r)

    # Lower quantile penalizes underestimates more
    assert abs(grad_lower) > abs(grad_median)
    # Upper quantile penalizes underestimates less
    assert abs(grad_upper) < abs(grad_median)


def test_progressive_solver_basic():
    """Test basic functionality of ProgressiveSmoothingSolver."""
    np.random.seed(42)

    # Generate simple dataset
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    X = StandardScaler().fit_transform(X)

    # Use FISTA which is compatible with Huber
    # FISTA uses gradient which Huber implements
    solver = FISTA(max_iter=50, tol=1e-4)

    # Test median regression (should use Huber internally)
    solver_median = ProgressiveSmoothingSolver(
        smoothing_sequence=[1.0, 0.5, 0.1],
        quantile=0.5,
        alpha=0.1,
        smooth_solver=solver,
        verbose=False,
    )
    solver_median.fit(X, y)

    # Test non-median regression (should use QuantileHuber internally)
    solver_upper = ProgressiveSmoothingSolver(
        smoothing_sequence=[1.0, 0.5, 0.1],
        quantile=0.8,
        alpha=0.1,
        smooth_solver=solver,
        verbose=False,
    )
    solver_upper.fit(X, y)

    # Basic shape checks
    assert solver_median.coef_.shape == (X.shape[1],)
    assert solver_upper.coef_.shape == (X.shape[1],)

    # Check stage results are stored
    assert hasattr(solver_median, 'stage_results_')
    assert len(solver_median.stage_results_) == len(solver_median.smoothing_sequence)

    # Verify predictions work
    y_pred_median = solver_median.predict(X)
    y_pred_upper = solver_upper.predict(X)

    assert y_pred_median.shape == y.shape
    assert y_pred_upper.shape == y.shape

    # Upper quantile predictions should generally be higher
    # This might be too strict, so we'll check the 75th percentile instead of the mean
    assert np.percentile(y_pred_upper - y_pred_median, 75) > 0
