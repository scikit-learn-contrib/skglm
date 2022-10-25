import numpy as np
import matplotlib.pyplot as plt
from group_lasso import LogisticGroupLasso

from skglm import SparseLogisticRegression

np.random.seed(0)
X = np.random.randn(6, 10)
y = np.ones(X.shape[0])
y[:len(y) // 2] = -1

X -= X.mean(keepdims=True)


alpha_max = np.max(np.abs(X.T @ y)) / (2 * len(y))

n_alphas = 75

me = np.zeros([n_alphas, X.shape[1]])
them = me.copy()

us = SparseLogisticRegression(
    alpha=alpha_max, fit_intercept=False, verbose=1, warm_start=True, tol=1e-10)
alphas = alpha_max * np.geomspace(1, 0.01, num=n_alphas)

for idx, alpha in enumerate(alphas):
    clf = LogisticGroupLasso(
        groups=np.arange(X.shape[1]), group_reg=alpha, l1_reg=0, fit_intercept=False,
        old_regularisation=False, supress_warning=True, tol=1e-10)

    clf.fit(X, y)
    them[idx] = clf.coef_[:, 1]
    us.alpha = alpha
    us.fit(X, y)
    me[idx] = us.coef_.squeeze()

fig, axarr = plt.subplots(1, 2, constrained_layout=True)
axarr[0].semilogx(alphas, me)
axarr[0].set_title("Regularization path skglm")
axarr[1].semilogx(alphas, them)
axarr[1].set_title("Regularization path yngvem")

axarr[1].set_xlabel("alpha")
axarr[0].set_xlabel("alpha")
plt.show(block=False)
