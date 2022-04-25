# data available at  https://www.dropbox.com/sh/32b3mr3xghi496g/AACNRS_NOsUXU-hrSLixNg0ja?dl=0


import time
from numpy.linalg import norm
import matplotlib.pyplot as plt
import numpy as np
from celer import GroupLasso
from skglm.solvers.gram import gram_group_lasso

X = np.load("design_matrix.npy")
y = np.load("target.npy")
groups = np.load("groups.npy")
weights = np.load("weights.npy")
grps = [list(np.where(groups == i)[0]) for i in range(1, 33)]


alpha_ratio = 1e-2
n_alphas = 10
tol = 1e-8

# Case 1: slower runtime for (very) small alphas
# alpha_max = 0.003471727067743962
alpha_max = np.max(np.linalg.norm((X.T @ y).reshape(-1, 5), axis=1)) / len(y)
alpha = alpha_max / 100
clf = GroupLasso(fit_intercept=False, tol=tol,
                 groups=grps, weights=weights, alpha=alpha, verbose=1)

t0 = time.time()
clf.fit(X, y)
t1 = time.time()

print(f"Celer: {t1 - t0:.3f} s")

t0 = time.time()
res = gram_group_lasso(X, y, alpha, groups=grps, tol=tol, weights=weights, max_iter=10_000,
                       check_freq=50)
t1 = time.time()

print(f"skglm gram: {t1 - t0:.3f} s")
