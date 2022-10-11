import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from skglm.utils import make_correlated_data
from skglm.experimental import ReweightedLasso

X, y, _ = make_correlated_data(n_samples=200, n_features=500, random_state=24)

n_samples, n_features = X.shape
alpha_max = norm(X.T @ y, ord=np.inf) / n_samples

alpha = alpha_max / 10

clf = ReweightedLasso(alpha=alpha, verbose=2, tol=1e-10)
clf.fit(X, y)

# consider that our solver has converged
w_star = clf.coef_
p_star = (norm(X @ w_star - y) ** 2) / (2 * n_samples) + alpha * np.sqrt(norm(w_star))

print(clf.loss_history)

# plt.close("all")
# plt.semilogy(np.arange(1, max_iter+1, obj_freq), np.array(objs) - p_star)
# plt.xlabel("CP iteration")
# plt.ylabel("$F(x) - F(x^*)$")
# plt.show(block=False)
