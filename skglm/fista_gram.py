import numpy as np
from numpy.linalg import norm
from numba import njit
from celer import Lasso
from benchopt.datasets.simulated import make_correlated_data
from skglm.utils import ST_vec


@njit
def primal(alpha, r, w, weights):
    p_obj = (r @ r) / (2 * len(r))
    return p_obj + alpha * np.sum(np.abs(w * weights))

@njit
def dual(alpha, norm_y2, theta, y):
    d_obj = - np.sum((y / (alpha * len(y)) - theta) ** 2)
    d_obj *= 0.5 * alpha ** 2 * len(y)
    d_obj += norm_y2 / (2 * len(y))
    return d_obj

@njit
def dnorm_l1(theta, X, weights):
    n_features = X.shape[1]
    scal = 0.
    for j in range(n_features):
        Xj_theta = X[:, j] @ theta
        scal = max(scal, Xj_theta / weights[j])
    return scal

@njit
def create_dual_point(r, alpha, X, weights):
    theta = r / (alpha * len(y))
    scal = dnorm_l1(theta, X, weights)
    if scal > 1.:
        theta /= scal
    return theta

@njit
def dual_gap(alpha, norm_y2, y, X, w, weights):
    r = y - X @ w
    p_obj = primal(alpha, r, w, weights)
    theta = create_dual_point(r, alpha, X, weights)
    d_obj = dual(alpha, norm_y2, theta, y)
    return p_obj, d_obj, p_obj - d_obj


n_samples, n_features = 1_000, 300
X, y, w_star = make_correlated_data(
    n_samples=n_samples, n_features=n_features, random_state=0)
alpha_max = norm(X.T @ y, ord=np.inf)

weights = np.random.normal(2, 0.4, n_features)

# Hyperparameters
max_iter = 1000
tol = 1e-8
reg = 0.1
group_size = 3
check_gap_freq = 100

alpha = alpha_max * reg / n_samples

L = np.linalg.norm(X, ord=2) ** 2 / n_samples

G = X.T @ X
Xty = X.T @ y

w = np.zeros(n_features)
z = np.zeros(n_features)

norm_y2 = y @ y

t_new = 1

for n_iter in range(max_iter):
    t_old = t_new
    t_new = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2
    w_old = w.copy()
    z -= (G @ z - Xty) / L / n_samples
    w = ST_vec(z, alpha / L * weights)
    z = w + (t_old - 1.) / t_new * (w - w_old)

    if n_iter % check_gap_freq == 0:
        p_obj, d_obj, gap = dual_gap(alpha, norm_y2, y, X, w, weights)
        print(f"iter {n_iter} :: p_obj {p_obj:.5f} :: d_obj {d_obj:.5f} " +
              f":: gap {gap:.5f}")
        if gap < tol:
            print("Convergence reached!")
            break

clf = Lasso(alpha, tol=tol, weights=weights, fit_intercept=False)
clf.fit(X, y)
np.testing.assert_allclose(w, clf.coef_, rtol=1e-5)
