import numpy as np
from skglm import GeneralizedLinearEstimator
from skglm.experimental.pdcd_ws import PDCD_WS
from skglm.experimental.quantile_regression import Pinball
from skglm.penalties import L1
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from skglm.utils.jit_compilation import compiled_clone


def generate_dummy_data(n_samples=1000, n_features=10, noise=0.1):
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise)
    return X, y


np.random.seed(42)

datafit = Pinball(0.5)
penalty = L1(alpha=0.1)
solver = PDCD_WS(
    max_iter=10,
    max_epochs=100,
    tol=1e-2,
    warm_start=False,
    verbose=1,
)

# estimator = GeneralizedLinearEstimator(
#     datafit=datafit,
#     penalty=penalty,
#     solver=solver,
# )

X, y = generate_dummy_data(
    n_samples=1000, # if this is reduced to 100 samples, it converges
    n_features=10,
)
# y -= y.mean()
# y += 0.1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

df = compiled_clone(datafit)
pen = compiled_clone(penalty)

res = solver.solve(X, y, df, pen)

# estimator.fit(X, y)
