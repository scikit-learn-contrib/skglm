"""
===========================================
Smooth Quantile Regression Example
===========================================

"""

import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import QuantileRegressor
from skglm.experimental.smooth_quantile_regressor import SmoothQuantileRegressor
from skglm.experimental.quantile_huber import QuantileHuber

X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
X = StandardScaler().fit_transform(X)
tau = 0.75

t0 = time.time()
reg_skglm = SmoothQuantileRegressor(quantile=tau).fit(X, y)
t1 = time.time()
reg_sklearn = QuantileRegressor(quantile=tau, alpha=0.1, solver='highs').fit(X, y)
t2 = time.time()

y_pred_skglm, y_pred_sklearn = reg_skglm.predict(X), reg_sklearn.predict(X)
coverage_skglm = np.mean(y <= y_pred_skglm)
coverage_sklearn = np.mean(y <= y_pred_sklearn)

print(f"\nTiming: skglm={t1-t0:.3f}s, sklearn={t2-t1:.3f}s, "
      f"speedup={(t2-t1)/(t1-t0):.1f}x")
print(f"Coverage (target {tau}): skglm={coverage_skglm:.3f}, "
      f"sklearn={coverage_sklearn:.3f}")
print(f"Non-zero coefs: skglm={np.sum(reg_skglm.coef_ != 0)}, "
      f"sklearn={np.sum(reg_sklearn.coef_ != 0)}")


# Visualizations
def pinball(y_true, y_pred):
    diff = y_true - y_pred
    return np.mean(np.where(diff >= 0, tau * diff, (1 - tau) * -diff))


print(f"Pinball loss: skglm={pinball(y, y_pred_skglm):.4f}, "
      f"sklearn={pinball(y, y_pred_sklearn):.4f}")

plt.figure(figsize=(12, 5))
plt.subplot(121)
residuals = np.linspace(-2, 2, 1000)
for delta in [1.0, 0.5, 0.1]:
    loss = QuantileHuber(quantile=tau, delta=delta)
    losses = [loss.value(np.array([r]), np.array([[1]]), np.array([0]))
              for r in residuals]
    plt.plot(residuals, losses, label=f'δ={delta}')
plt.plot(residuals, [tau * max(r, 0) + (1 - tau) * max(-r, 0)
                     for r in residuals], 'k--', label='Pinball')
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.xlabel('Residual (y - y_pred)')
plt.ylabel('Loss')
plt.title('Quantile Huber Loss (τ=0.75)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(122)
plt.hist(y - y_pred_skglm, bins=50, alpha=0.5, label='skglm')
plt.hist(y - y_pred_sklearn, bins=50, alpha=0.5, label='sklearn')
plt.axvline(0, color='k', linestyle='--')
plt.xlabel('Residual (y - y_pred)')
plt.ylabel('Count')
plt.title('Residuals Histogram')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
