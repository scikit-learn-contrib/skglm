from time import time
import numpy as np
from numpy.linalg import norm
from celer import Lasso, GroupLasso
from benchopt.datasets.simulated import make_correlated_data
from skglm.solvers.gram import gram_fista_group_lasso, gram_fista_lasso, gram_lasso, gram_group_lasso


n_samples, n_features = 100, 300
X, y, w_star = make_correlated_data(
    n_samples=n_samples, n_features=n_features, random_state=0)
alpha_max = norm(X.T @ y, ord=np.inf)

# Hyperparameters
max_iter = 1000
tol = 1e-8
reg = 0.1
group_size = 3

alpha = alpha_max * reg / n_samples

weights = np.random.normal(2, 0.4, n_features)
weights_grp = np.random.normal(2, 0.4, n_features // group_size)

# Lasso
# print("#" * 15)
# print("Lasso")
# print("#" * 15)
# start = time()
# w = gram_lasso(X, y, alpha, max_iter, tol, weights=weights)
# gram_lasso_time = time() - start
# clf_sk = Lasso(alpha, weights=weights, tol=tol, fit_intercept=False)
# start = time()
# clf_sk.fit(X, y)
# celer_lasso_time = time() - start
# start = time()
# w_fista = gram_fista_lasso(X, y, alpha, max_iter, tol, weights=weights)
# gram_fista_lasso_time = time() - start
# np.testing.assert_allclose(w, clf_sk.coef_, rtol=1e-4)
# np.testing.assert_allclose(w, w_fista, rtol=1e-4)

# print("\n")
# print("Celer: %.2f" % celer_lasso_time)
# print("Gram: %.2f" % gram_lasso_time)
# print("FISTA Gram: %.2f" % gram_fista_lasso_time)
# print("\n")

# Group Lasso
print("#" * 15)
print("Group Lasso")
print("#" * 15)
start = time()
w = gram_group_lasso(X, y, alpha, group_size, max_iter, tol, weights=weights_grp)
gram_group_lasso_time = time() - start
start = time()
w_fista = gram_fista_group_lasso(X, y, alpha, group_size, max_iter, tol, 
                                 weights=weights_grp)
gram_fista_group_lasso_time = time() - start

np.testing.assert_allclose(w, w_fista, rtol=1e-4)

# clf_celer = GroupLasso(group_size, alpha, tol=tol, weights=weights_grp, 
#                        fit_intercept=False)
# start = time()
# clf_celer.fit(X, y)
# celer_group_lasso_time = time() - start
# np.testing.assert_allclose(w, clf_celer.coef_, rtol=1e-1)

print("\n")
# print("Celer: %.2f" % celer_group_lasso_time)
print("Gram: %.2f" % gram_group_lasso_time)
print("FISTA Gram: %.2f" % gram_fista_group_lasso_time)
print("\n")
