from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_regression
from skglm import GeneralizedLinearEstimator
from skglm.datafits import Quadratic
from skglm.penalties import L1
from skglm.solvers import AndersonCD


def test_gridsearch_compatibility():
    # Generate synthetic data
    X, y = make_regression(n_samples=100, n_features=10, random_state=42)

    # Create base estimator
    base_estimator = GeneralizedLinearEstimator(
        datafit=Quadratic(),
        penalty=L1(1.0),
        solver=AndersonCD()
    )

    # Define parameter grid
    param_grid = {
        'penalty__alpha': [0.1, 1.0, 10.0],
        'solver__max_iter': [10, 20],
        'solver__tol': [1e-3, 1e-4]
    }

    # Create GridSearchCV
    grid_search = GridSearchCV(
        base_estimator,
        param_grid,
        cv=3,
        scoring='neg_mean_squared_error'
    )

    # Fit GridSearchCV
    grid_search.fit(X, y)

    # Verify that GridSearchCV worked
    assert hasattr(grid_search, 'best_params_')
    assert hasattr(grid_search, 'best_score_')
    assert hasattr(grid_search, 'best_estimator_')

    # Verify that best_estimator_ has the correct parameters
    best_estimator = grid_search.best_estimator_
    assert isinstance(best_estimator, GeneralizedLinearEstimator)
    assert best_estimator.penalty.alpha in [0.1, 1.0, 10.0]
    assert best_estimator.solver.max_iter in [10, 20]
    assert best_estimator.solver.tol in [1e-3, 1e-4]
