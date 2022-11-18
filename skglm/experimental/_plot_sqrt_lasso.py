
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from skglm.utils.data import make_correlated_data
from skglm.experimental.sqrt_lasso import SqrtLasso, _chambolle_pock_sqrt

X, y, _ = make_correlated_data(n_samples=200, n_features=100, random_state=24)

n_samples, n_features = X.shape
alpha_max = norm(X.T @ y, ord=np.inf) / (norm(y) * np.sqrt(n_samples))

alpha = alpha_max / 10


max_iter = 1000
obj_freq = 10
w, _, objs = _chambolle_pock_sqrt(X, y, alpha, max_iter=max_iter, obj_freq=obj_freq)


# no convergence issue if n_features < n_samples, can use ProxNewton
# clf = SqrtLasso(alpha=alpha / np.sqrt(n_samples), verbose=2, tol=1e-10)
clf = SqrtLasso(alpha=alpha, verbose=2, tol=1e-10)
clf.fit(X, y)

# consider that our solver has converged
w_star = clf.coef_
p_star = norm(X @ w_star - y) / np.sqrt(n_samples) + alpha * norm(w_star, ord=1)

plt.close("all")
plt.semilogy(np.arange(1, max_iter+1, obj_freq), np.array(objs) - p_star)
plt.xlabel("CP iteration")
plt.ylabel("$F(x) - F(x^*)$")
plt.show(block=False)
