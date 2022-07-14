import numpy as np
import time
from skglm.estimators import MultiTaskLasso

from skglm.datafits import QuadraticMultiTask
from skglm.penalties import L2_1
from skglm.solvers.multitask_bcd_solver import bcd_solver 


n_samples, n_features, n_tasks = 30, 100, 10

X = np.random.normal(0, 1, (n_samples, n_features))
Y = np.random.normal(0, 1, (n_samples, n_tasks))
alpha_max = np.max(np.linalg.norm(X.T @ Y, ord=2, axis=1)) / n_samples
alpha = alpha_max / 10

# compil:
clf = MultiTaskLasso(alpha=alpha, max_iter=1, fit_intercept=False, verbose=1).fit(X, Y)
t0 = time.time()
clf = MultiTaskLasso(alpha=alpha, max_iter=3, fit_intercept=False, verbose=1).fit(X, Y)
t1 = time.time()
print(f"Fit with max_iter=3: {(t1 -t0):.2f} s")

print("#" * 80)
quad = QuadraticMultiTask()
quad.initialize(X, Y)
t0 = time.time()
quad.initialize(X, Y)
t1 = time.time()
print(f"quad init aka ~2x KKT/Grad computation: {(t1 -t0):.2f} s")
print("#" * 80)


pen = L2_1(alpha=alpha)

# W = np.zeros((n_features, n_tasks))
# XW = np.zeros((n_samples, n_tasks))
# t0 = time.time()
# bcd_solver(X, Y, quad, pen, W, XW, max_iter=3, verbose=2, use_acc=False)
# t1 = time.time()
# print(f"solver call: {t1 - t0:.2f} s")


W = np.zeros((n_features, n_tasks))
XW = np.zeros((n_samples, n_tasks))
bcd_solver(X, Y, quad, pen, W, XW, max_iter=3, use_acc=False)
