import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from skglm.utils.jit_compilation import compiled_clone
from skglm.datafits import Logistic
from skglm.penalties import MCPenalty, L1

from skglm.solvers import ProxNewton, AndersonCD

from benchopt.datasets import make_correlated_data


X_, y_, _ = make_correlated_data(n_samples=400, n_features=500, random_state=0)
y_ = np.sign(y_) / 2 + .5

X_train, X_test, y_train, y_test = train_test_split(X_, y_, random_state=0)

alpha_max = norm(X_train.T @ y_train, np.inf) / (2 * len(y_train))

df = compiled_clone(Logistic())
df.initialize(X_train, y_train)

alphas = np.geomspace(alpha_max, alpha_max / 100, num=50)
errs = np.zeros_like(alphas)
supp_size = np.zeros_like(alphas)
w = np.zeros(X_train.shape[1])
for idx, alpha in enumerate(alphas):
    # pen_l1 = compiled_clone(L1(alpha=alpha))
    pen_l1 = compiled_clone(MCPenalty(alpha=alpha, gamma=3))
    solver = ProxNewton(verbose=1, fit_intercept=False)
    # w = solver.solve(X_train, y_train, df, pen_l1)[0] #, w_init=w.copy())[0]
    w = solver.solve(X_train, y_train, df, pen_l1, w_init=w.copy())[0]
    y_pred = (X_test @ w > 0).astype(int)
    errs[idx] = f1_score(y_test, y_pred)
    supp_size[idx] = (w != 0).sum()


# plt.close('all')
fig, axarr = plt.subplots(2, 1, sharex=True)
ax = axarr[0]
ax.set_ylabel("F1 score on left out data")
ax.semilogx(alphas / alpha_max, errs)
ax.set_ylim(0, 1)
ax = axarr[1]
ax.semilogy(alphas / alpha_max, supp_size)
ax.set_ylabel("support size")
plt.show(block=False)