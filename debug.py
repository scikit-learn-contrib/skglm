# %%
import numpy as np
from skglm import GeneralizedLinearEstimator
from skglm.experimental.pdcd_ws import PDCD_WS
from skglm.experimental.quantile_regression import Pinball
from skglm.penalties import L1
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from skglm.utils.jit_compilation import compiled_clone
from sklearn.linear_model import QuantileRegressor


def generate_dummy_data(n_samples=1000, n_features=10, noise=0.1):
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise)
    # y -= y.mean()
    # y += 0.1
    y /= 10
    return X, y


np.random.seed(42)

quantile_level = 0.5
alpha = 0.1

X, y = generate_dummy_data(
    n_samples=1000,  # if this is reduced to 100 samples, it converges
    n_features=11,
)

solver = PDCD_WS(
    p0=11,
    max_iter=50,
    max_epochs=500,
    tol=1e-5,
    warm_start=False,
    verbose=2,
)

datafit = Pinball(quantile_level)
penalty = L1(alpha=alpha)

df = compiled_clone(datafit)
pen = compiled_clone(penalty)

res = solver.solve(X, y, df, pen)

# %%

clf = QuantileRegressor(
    quantile=quantile_level,
    alpha=alpha/len(y),
    fit_intercept=False,
    solver='highs',
).fit(X, y)

# %%
print("diff solution:", np.linalg.norm((clf.coef_ - res[0])))

# %%


def obj_val(w):
    return df.value(y, w, X @ w) + pen.value(w)


for label, w in zip(("skglm", "sklearn"), (res[0], clf.coef_)):
    print(f"{label:10} {obj_val(w)=}")

# %%
