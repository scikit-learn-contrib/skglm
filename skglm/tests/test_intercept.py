import numpy as np
import pytest
from sklearn.linear_model import enet_path
from sklearn.linear_model import MultiTaskLasso as MultiTaskLasso_sklearn
from sklearn.linear_model import HuberRegressor, Lasso, LogisticRegression

from skglm.solvers.cd_solver import cd_solver, cd_solver_path
from skglm.solvers.multitask_bcd_solver import multitask_bcd_solver
from skglm.datafits import Quadratic, Logistic, Huber, QuadraticMultiTask
from skglm.penalties import L1, WeightedL1, L2_1
from skglm.utils import make_correlated_data
from skglm.utils import compiled_clone

X, Y, _ = make_correlated_data(n_samples=20, n_features=10, random_state=0, n_tasks=9)
n_samples, n_features = X.shape
# Lasso will fit with binary values, but else logreg's alpha_max is wrong:
y = Y[:, 0]
y = np.sign(y)
alpha_max = np.linalg.norm(X.T @ y, ord=np.inf) / n_samples
alpha = 0.05 * alpha_max

tol = 1e-14

dict_estimator_sklearn = {}
dict_estimator_sklearn["Lasso"] = Lasso(alpha=alpha, fit_intercept=True, tol=tol)
dict_estimator_sklearn["Logistic"] = LogisticRegression(
    penalty='l1', C=1/(alpha * len(y)), tol=tol, solver='saga',
    fit_intercept=True, max_iter=10_000)
dict_estimator_sklearn["Huber"] = HuberRegressor(
    fit_intercept=True, alpha=0, tol=tol, epsilon=1.35,
    max_iter=10_000).fit(X, y)

dict_estimator = {}
dict_estimator["Lasso"] = (Quadratic(), L1(alpha))
dict_estimator["Logistic"] = (Logistic(), L1(alpha))

delta = dict_estimator_sklearn["Huber"].epsilon * dict_estimator_sklearn["Huber"].scale_
dict_estimator["Huber"] = (Huber(delta), WeightedL1(1, np.zeros(X.shape[1])))


@pytest.mark.parametrize(
    "name_estimator",
    ["Lasso", "Logistic", "Huber"])
def test_intercept(name_estimator):
    # initialize datafits
    datafit, penalty = dict_estimator[name_estimator]
    penalty = compiled_clone(penalty)
    datafit = compiled_clone(datafit, to_float32=X.dtype == np.float32)

    datafit.initialize(X, y)

    # initialize coefficients for cd solver
    w = np.zeros(X.shape[1] + 1)
    Xw = np.zeros(X.shape[0])
    coefs_ours, _, _ = cd_solver(
        X, y, datafit, penalty,
        w, Xw, fit_intercept=True, max_iter=50,
        max_epochs=50_000, tol=tol, verbose=0)

    intercept_ours = np.array([coefs_ours[-1]])
    coefs_ours = coefs_ours[:X.shape[1]]
    coefs_sk = dict_estimator_sklearn[name_estimator].fit(X, y).coef_

    intercept_sk = dict_estimator_sklearn[name_estimator].intercept_
    np.testing.assert_allclose(coefs_ours, coefs_sk.flatten(), atol=1e-4)
    np.testing.assert_allclose(intercept_sk, intercept_ours, atol=1e-4)


# Test if skglm multitask solver returns the coefficients
def test_intercept_mtl():
    # initialize datafits
    datafit, penalty = QuadraticMultiTask(), L2_1(alpha=alpha)
    penalty = compiled_clone(penalty)
    datafit = compiled_clone(datafit, to_float32=X.dtype == np.float32)

    datafit.initialize(X, Y)

    # initialize coefficients for cd solver
    W = np.zeros((X.shape[1] + 1, Y.shape[1]))
    XW = np.zeros((X.shape[0], Y.shape[1]))
    coefs_ours, _, _ = multitask_bcd_solver(
        X, Y, datafit, penalty,
        W, XW, fit_intercept=True, max_iter=50,
        max_epochs=50_000, tol=tol, verbose=0, use_acc=True)

    intercept_ours = coefs_ours[-1, :]
    coefs_ours = coefs_ours[:X.shape[1], :]
    mlt_sk = MultiTaskLasso_sklearn(alpha=alpha, fit_intercept=True, tol=1e-8)
    coefs_sk = mlt_sk.fit(X, Y).coef_
    intercept_sk = mlt_sk.intercept_
    np.testing.assert_allclose(coefs_ours.T, coefs_sk, atol=1e-4)
    np.testing.assert_allclose(intercept_sk, intercept_ours, atol=1e-4)


def test_path():
    n_samples, n_features = 10, 20
    X, y, _ = make_correlated_data(n_samples, n_features, random_state=0)
    alpha_max = np.linalg.norm(X.T @ y, ord=np.inf) / n_samples

    quad_datafit = compiled_clone(Quadratic())
    l1_penalty = compiled_clone(L1(alpha=1.))
    alphas = np.geomspace(alpha_max, alpha_max / 10, num=3)

    _, coefs, _ = cd_solver_path(
        X, y, quad_datafit, l1_penalty, alphas=alphas, tol=1e-10)
    _, sk_coefs, _ = enet_path(X, y, l1_ratio=1., alphas=alphas, tol=1e-10)
    np.testing.assert_allclose(coefs, sk_coefs, atol=1e-5)


if __name__ == '__main__':
    test_intercept_mtl()
