from skglm.solvers.cd_solver import cd_solver
from skglm.datafits import Quadratic, Logistic, Huber
from skglm.penalties import L1, WeightedL1
from skglm.utils import make_correlated_data
from sklearn.linear_model import HuberRegressor, Lasso, LogisticRegression
import numpy as np
import pytest
from skglm.utils import compiled_clone

X, y, _ = make_correlated_data(n_samples=20, n_features=10, random_state=0)
n_samples, n_features = X.shape
# Lasso will fit with binary values, but else logreg's alpha_max is wrong:
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
            max_epochs=50_000, tol=tol, verbose=0, use_acc=True)

    intercept_ours = np.array([coefs_ours[-1]])
    coefs_ours = coefs_ours[:X.shape[1]]
    coefs_sk = dict_estimator_sklearn[name_estimator].fit(X, y).coef_

    intercept_sk = dict_estimator_sklearn[name_estimator].intercept_
    np.testing.assert_allclose(coefs_ours, coefs_sk.flatten(), atol=1e-4)
    np.testing.assert_allclose(intercept_sk, intercept_ours, atol=1e-4)
