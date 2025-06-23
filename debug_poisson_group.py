import numpy as np
from skglm import GeneralizedLinearEstimator
from skglm.datafits.group import PoissonGroup
from skglm.penalties import WeightedGroupL2
from skglm.solvers import GroupProxNewton
from sklearn.metrics import mean_squared_error

# Sample data and group structure
n_samples, n_features = 20, 10
X = np.random.randn(n_samples, n_features)
y = np.random.poisson(np.abs(X[:, 0] + X[:, 5]))

grp_ptr = np.array([0, 3, 5, 8, 10], dtype=np.int32)
grp_indices = np.arange(n_features, dtype=np.int32)

# Estimator setup
estimator = GeneralizedLinearEstimator(
    datafit=PoissonGroup(grp_ptr, grp_indices),
    penalty=WeightedGroupL2(alpha=0.1, grp_ptr=grp_ptr, grp_indices=grp_indices,
                            weights=np.ones(len(grp_ptr) - 1)),
    solver=GroupProxNewton()
)

estimator.fit(X, y)
print("Coefficients:", estimator.coef_)
print("Intercept:", estimator.intercept_)
y_pred = estimator.predict(X)
print("First 5 predictions:", y_pred[:5])
print("First 5 true values:", y[:5])
print("MSE:", mean_squared_error(y, np.exp(y_pred)))
