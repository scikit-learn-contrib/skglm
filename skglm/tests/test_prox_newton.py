import pytest
import numpy as np
from numpy.linalg import norm

from sklearn.linear_model import LogisticRegression as LogReg_sklearn

from skglm.penalties import L1, L1_plus_L2
from skglm.datafits import Logistic
from skglm.solvers.cd_solver import cd_solver
from skglm.solvers.prox_newton_solver import prox_newton_solver

from skglm.utils import make_correlated_data


n_samples, n_features = 30, 50
X, y, _ = make_correlated_data(
    n_samples=n_samples, n_features=n_features, density=0.1, random_state=0)
y_ind = np.sign(y)

np.random.seed(0)

alpha_max = norm(X.T @ y, ord=np.inf) / (2 * n_samples)
alpha = alpha_max * 0.05
tol = 1e-10
l1_ratio = 0.3

dict_penalties = {}

dict_penalties["L1"] = L1(alpha=alpha)
dict_penalties["L1_plus_L2"] = L1_plus_L2(alpha=alpha, l1_ratio=l1_ratio)


def test_prox_newton_vs_sklearn():
    datafit = Logistic()
    datafit.initialize(X, y_ind)
    pen = L1(alpha=alpha)
    w = np.zeros(n_features)
    Xw = np.zeros(n_samples)
    w_newton = prox_newton_solver(X, y_ind, datafit, pen, w, Xw, tol=tol)[0]

    estimator_sk = LogReg_sklearn(
        C=1/(alpha * n_samples), fit_intercept=False, tol=tol, penalty='l1',
        solver='liblinear')
    estimator_sk.fit(X, y_ind)

    np.testing.assert_allclose(w_newton, np.ravel(estimator_sk.coef_), atol=1e-5)


@pytest.mark.parametrize("penalty_name", ["L1", "L1_plus_L2"])
def test_prox_newton_vs_cd(penalty_name):
    datafit = Logistic()
    datafit.initialize(X, y_ind)
    pen = dict_penalties[penalty_name]

    w = np.zeros(n_features)
    Xw = np.zeros(n_samples)
    w_newton = prox_newton_solver(X, y_ind, datafit, pen, w, Xw, tol=tol)[0]

    w = np.zeros(n_features)
    Xw = np.zeros(n_samples)
    w_cd = cd_solver(X, y_ind, datafit, pen, w, Xw, tol=tol)[0]

    np.testing.assert_allclose(w_newton, w_cd, atol=1e-5)


if __name__ == '__main__':
    # LOGGER in action
    datafit = Logistic()
    datafit.initialize(X, y_ind)
    pen = L1(alpha=alpha)
    w = np.zeros(n_features)
    Xw = np.zeros(n_samples)
    w_newton = prox_newton_solver(X, y_ind, datafit, pen, w, Xw, tol=tol)[0]
