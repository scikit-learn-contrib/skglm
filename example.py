import time
import numpy as np
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import QuantileRegressor
from skglm.experimental.smooth_quantile_regressor import SmoothQuantileRegressor
from sklearn.model_selection import train_test_split

from numpy.linalg import norm


def pinball_loss(y_true, y_pred, tau=0.5):
    """Compute Pinball (quantile) loss."""
    residuals = y_true - y_pred
    return np.mean(np.where(residuals >= 0,
                            tau * residuals,
                            (1 - tau) * -residuals))


# Test different problem sizes
n_samples, n_features = 100, 100
X, y = make_regression(n_samples=n_samples, n_features=n_features,
                       noise=0.1, random_state=0)
alpha = 0.01

# Test different noise distributions

# Quantiles to test
tau = 0.3

# Store results
results = []

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# scikit-learn QuantileRegressor
qr = QuantileRegressor(quantile=tau, alpha=alpha, solver="highs")
t0 = time.time()
qr.fit(X_train, y_train)
qr_time = time.time() - t0


ours = SmoothQuantileRegressor(quantile=tau, alpha=alpha)
t0 = time.time()
ours.fit(X_train, y_train)
ours_time = time.time() - t0


print(ours.coef_ - qr.coef_)
print(norm(ours.coef_ - qr.coef_) / norm(qr.coef_))
