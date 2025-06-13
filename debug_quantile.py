"""
QuantileHuber vs Sklearn
"""
import numpy as np
import time
from sklearn.linear_model import QuantileRegressor
from skglm.experimental.quantile_huber import QuantileHuber, SmoothQuantileRegressor
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression


def pinball_loss(residuals, quantile):
    """True pinball loss."""
    return np.mean(residuals * (quantile - (residuals < 0)))


X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=0)
tau = 0.8

start = time.time()
sk = QuantileRegressor(quantile=tau, alpha=0.1, fit_intercept=True)
sk.fit(X, y)
sk_pred = sk.predict(X)
sk_time = time.time() - start
sk_cov = np.mean(y <= sk_pred)
sk_pinball = pinball_loss(y - sk_pred, tau)

start = time.time()
qh = SmoothQuantileRegressor(
    quantile=tau,
    alpha=0.1,
    delta_init=0.5,
    delta_final=0.001,
    n_deltas=5,
    solver="AndersonCD",
    verbose=True,
    fit_intercept=True
)
qh.fit(X, y)
qh_time = time.time() - start
qh_pred = qh.predict(X)
qh_cov = np.mean(y <= qh_pred)
qh_pinball = pinball_loss(y - qh_pred, tau)


print(sk.coef_)
print(qh.est.coef_)

# print(f"{'Method':<12} {'Q':<4} {'Coverage':<8} {'Time':<6} "
#       f"{'Pinball':<8}")
# print("-" * 55)
# print(f"{'Sklearn':<12} {tau:<4} {sk_cov:<8.3f} {sk_time:<6.3f} "
#       f"{sk_pinball:<8.4f}")
# print(f"{'QuantileHuber':<12} {tau:<4} {qh_cov:<8.3f} {qh_time:<6.3f} "
#       f"{qh_pinball:<8.4f}")


# quantiles = [0.1, 0.5, 0.9]
# delta = 0.5
# residuals = np.linspace(-3, 3, 500)
# _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
# for tau in quantiles:
#     qh = QuantileHuber(quantile=tau, delta=delta)
#     loss = [qh._loss_sample(r) for r in residuals]
#     grad = [qh._grad_per_sample(r) for r in residuals]
#     ax1.plot(residuals, loss, label=f"τ={tau}")
#     ax2.plot(residuals, grad, label=f"τ={tau}")
# ax1.set_title("QuantileHuber Loss")
# ax1.set_xlabel("Residual")
# ax1.set_ylabel("Loss")
# ax1.legend()
# ax2.set_title("QuantileHuber Gradient")
# ax2.set_xlabel("Residual")
# ax2.set_ylabel("Gradient")
# ax2.legend()
# plt.tight_layout()
# plt.show()
