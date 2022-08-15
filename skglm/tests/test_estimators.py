from collections.abc import Iterable

import pytest
import numpy as np
from numpy.linalg import norm

from sklearn.base import copy
from sklearn.linear_model import Lasso as Lasso_sklearn
from sklearn.linear_model import ElasticNet as ElasticNet_sklearn
from sklearn.linear_model import LogisticRegression as LogReg_sklearn
from sklearn.linear_model import MultiTaskLasso as MultiTaskLasso_sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC as LinearSVC_sklearn
from sklearn.utils.estimator_checks import check_estimator

from scipy.sparse import csc_matrix, issparse

from skglm.utils import make_correlated_data
from skglm.estimators import (
    GeneralizedLinearEstimator, Lasso, MultiTaskLasso, WeightedLasso, ElasticNet,
    MCPRegression, SparseLogisticRegression, LinearSVC)
from skglm.datafits import Logistic, Quadratic, QuadraticSVC, QuadraticMultiTask
from skglm.penalties import L1, IndicatorBox, L1_plus_L2, MCPenalty, WeightedL1


n_samples = 50
n_tasks = 9
n_features = 60
X, Y, _ = make_correlated_data(
    n_samples=n_samples, n_features=n_features, n_tasks=n_tasks, density=0.1,
    random_state=0)
y = Y[:, 0]

np.random.seed(0)
X_sparse = csc_matrix(X * np.random.binomial(1, 0.1, X.shape))

n_samples, n_features = X.shape
# Lasso will fit with binary values, but else logreg's alpha_max is wrong:
y = np.sign(y)
alpha_max = norm(X.T @ y, ord=np.inf) / n_samples
alpha = 0.05 * alpha_max
C = 1 / alpha
tol = 1e-10
l1_ratio = 0.3

dict_estimators_sk = {}
dict_estimators_ours = {}

dict_estimators_sk["Lasso"] = Lasso_sklearn(
    alpha=alpha, fit_intercept=False, tol=tol)
dict_estimators_ours["Lasso"] = Lasso(
    alpha=alpha, fit_intercept=False, tol=tol)

dict_estimators_sk["wLasso"] = Lasso_sklearn(
    alpha=alpha, fit_intercept=False, tol=tol)
dict_estimators_ours["wLasso"] = WeightedLasso(
    alpha=alpha, fit_intercept=False, tol=tol, weights=np.ones(n_features))

dict_estimators_sk["ElasticNet"] = ElasticNet_sklearn(
    alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False, tol=tol)
dict_estimators_ours["ElasticNet"] = ElasticNet(
    alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False, tol=tol)

dict_estimators_sk["MCP"] = Lasso_sklearn(
    alpha=alpha, fit_intercept=False, tol=tol)
dict_estimators_ours["MCP"] = MCPRegression(
    alpha=alpha, gamma=np.inf, fit_intercept=False, tol=tol)

dict_estimators_sk["LogisticRegression"] = LogReg_sklearn(
    C=1/(alpha * n_samples), fit_intercept=False, tol=tol, penalty='l1',
    solver='liblinear')
dict_estimators_ours["LogisticRegression"] = SparseLogisticRegression(
    alpha=alpha, fit_intercept=False, tol=tol, verbose=True)

C = 1.0
dict_estimators_sk["SVC"] = LinearSVC_sklearn(
    penalty='l2', loss='hinge', fit_intercept=False, dual=True, C=C, tol=tol)
dict_estimators_ours["SVC"] = LinearSVC(C=C, tol=tol)


# Currently, `GeneralizedLinearEstimator` does not pass sklearn's `check_estimator`
# tests. Indeed, jitclasses which `GeneralizedLinearEstimator` depends upon (both the
# datafit and penalty objects are jitclasses) are not serializable ("pickleable"). It is
# a non-trivial well-known issue in Numba.
# For more information, see: https://github.com/numba/numba/issues/1846 .
@pytest.mark.parametrize(
    "estimator_name",
    ["Lasso", "wLasso", "ElasticNet", "MCP", "LogisticRegression", "SVC"])
def test_check_estimator(estimator_name):
    clf = copy.copy(dict_estimators_ours[estimator_name])
    clf.tol = 1e-6  # failure in float32 computation otherwise
    if isinstance(clf, WeightedLasso):
        clf.weights = None
    check_estimator(clf)


@pytest.mark.parametrize("estimator_name", dict_estimators_ours.keys())
@pytest.mark.parametrize('X', [X, X_sparse])
def test_estimator(estimator_name, X):
    if estimator_name == "GeneralizedLinearEstimator":
        pytest.skip()
    estimator_sk = dict_estimators_sk[estimator_name]
    estimator_ours = dict_estimators_ours[estimator_name]

    estimator_sk.fit(X, y)
    estimator_ours.fit(X, y)
    coef_sk = estimator_sk.coef_
    coef_ours = estimator_ours.coef_
    # assert that something was fitted:
    np.testing.assert_array_less(1e-5, norm(coef_ours))
    np.testing.assert_allclose(coef_ours, coef_sk, atol=1e-6)


@pytest.mark.parametrize('X', [X, X_sparse])
def test_estimator_mtl(X):
    estimator_sk = MultiTaskLasso_sklearn(
        alpha, fit_intercept=False, tol=1e-8)
    estimator_ours = MultiTaskLasso(
        alpha, verbose=2, max_iter=10, fit_intercept=False, tol=1e-8)

    estimator_sk.fit(X.toarray() if issparse(X) else X, Y)
    estimator_ours.fit(X, Y)
    coef_sk = estimator_sk.coef_
    coef_ours = estimator_ours.coef_
    np.testing.assert_allclose(coef_ours, coef_sk, atol=1e-6)


def test_mtl_path():
    alphas = np.geomspace(alpha_max, alpha_max * 0.01, 10)
    alpha_sk, coef_sk, _ = MultiTaskLasso_sklearn(
        alpha, fit_intercept=False).path(
            X, Y, l1_ratio=1, tol=1e-10, max_iter=5_000, alphas=alphas)
    coef_ours = MultiTaskLasso(alpha_max, fit_intercept=False).path(
        X, Y, alpha_sk, max_iter=10, tol=1e-10)[1]
    np.testing.assert_allclose(coef_ours, coef_sk, rtol=1e-5)


@pytest.mark.parametrize("Datafit, Penalty, is_classif, Estimator, pen_args", [
    (Quadratic, L1, False, Lasso, [alpha]),
    (Quadratic, WeightedL1, False, WeightedLasso,
     [alpha, np.random.choice(3, n_features)]),
    (Quadratic, L1_plus_L2, False, ElasticNet, [alpha, 0.3]),
    (Quadratic, MCPenalty, False, MCPRegression, [alpha, 3]),
    (QuadraticSVC, IndicatorBox, True, LinearSVC, [alpha]),
    (Logistic, L1, True, SparseLogisticRegression, [alpha]),
])
def test_generic_estimator(Datafit, Penalty, is_classif, Estimator, pen_args):
    target = Y if Datafit == QuadraticMultiTask else y
    clf = GeneralizedLinearEstimator(Datafit(), Penalty(*pen_args), is_classif,
                                     tol=1e-10, fit_intercept=False).fit(X, target)
    clf_est = Estimator(*pen_args, tol=1e-10, fit_intercept=False).fit(X, target)
    np.testing.assert_allclose(clf_est.coef_, clf.coef_, rtol=1e-5)


@pytest.mark.parametrize("Datafit, Penalty, Estimator_sk", [
    (Logistic, L1, LogReg_sklearn),
    (Quadratic, L1, Lasso_sklearn),
    (QuadraticSVC, IndicatorBox, LinearSVC_sklearn)
])
def test_estimator_predict(Datafit, Penalty, Estimator_sk):
    is_classif = Datafit in (Logistic, QuadraticSVC)
    if y.dtype.type == np.str_ and not is_classif:
        pytest.skip()
    estim_args = {
        LogReg_sklearn: {
            "C": 1 / n_samples, "tol": tol, "solver": "liblinear", "penalty": "l1",
            "fit_intercept": False},
        LinearSVC_sklearn: {
            "penalty": 'l2', "loss": 'hinge', "fit_intercept": False, "dual": True,
            "C": 1., "tol": tol},
        Lasso_sklearn: {"alpha": 1., "fit_intercept": False, "tol": tol}
    }
    X_test = np.random.normal(0, 1, (n_samples, n_features))
    clf = GeneralizedLinearEstimator(
        Datafit(), Penalty(1.), is_classif, fit_intercept=False, tol=tol).fit(X, y)
    clf_sk = Estimator_sk(**estim_args[Estimator_sk]).fit(X, y)
    y_pred = clf.predict(X_test)
    y_pred_sk = clf_sk.predict(X_test)
    if is_classif:
        np.testing.assert_equal(y_pred, y_pred_sk)
    else:
        np.testing.assert_allclose(y_pred, y_pred_sk, rtol=1e-5)


def test_generic_get_params():
    def assert_deep_dict_equal(expected_attr, estimator):
        """Helper function for deep equality in dictionary. Straight == fails."""
        for k, v in expected_attr.items():
            v_est = estimator.get_params()[k]
            if isinstance(v, Iterable):
                np.testing.assert_allclose(v, v_est)
            else:
                assert v == v_est

    reg = GeneralizedLinearEstimator(Quadratic(), L1(4.), is_classif=False)
    clf = GeneralizedLinearEstimator(Logistic(), MCPenalty(2., 3.), is_classif=True)

    # Xty and lipschitz attributes are defined for jit compiled classes
    # hence they are not included in the test
    expected_clf_attr = {'penalty__alpha': 2., 'penalty__gamma': 3.}
    expected_reg_attr = {'penalty__alpha': 4.}
    assert_deep_dict_equal(expected_reg_attr, reg)
    assert_deep_dict_equal(expected_clf_attr, clf)


# LogisticRegression and SVC can't be tested since sklearn adopts another name for
# the regularization parameter (`C` for sklearn, `alpha` in skglm).
@pytest.mark.parametrize(
    "estimator_name",
    ["Lasso", "wLasso", "ElasticNet", "MCP"])
def test_grid_search(estimator_name):
    estimator_sk = dict_estimators_sk[estimator_name]
    estimator_ours = dict_estimators_ours[estimator_name]
    estimator_sk.tol = 1e-10
    estimator_ours.tol = 1e-10
    estimator_sk.max_iter = 5000
    estimator_ours.max_iter = 100
    param_grid = {'alpha': np.geomspace(alpha_max, alpha_max * 0.01, 10)}
    sk_clf = GridSearchCV(estimator_sk, param_grid).fit(X, y)
    ours_clf = GridSearchCV(estimator_ours, param_grid).fit(X, y)
    res_attr = ["split%i_test_score" % i for i in range(5)] + \
               ["mean_test_score", "std_test_score", "rank_test_score"]
    for attr in res_attr:
        np.testing.assert_allclose(sk_clf.cv_results_[attr], ours_clf.cv_results_[attr],
                                   rtol=1e-3)
    np.testing.assert_allclose(sk_clf.best_score_, ours_clf.best_score_, rtol=1e-3)
    np.testing.assert_allclose(sk_clf.best_params_["alpha"],
                               ours_clf.best_params_["alpha"], rtol=1e-3)


if __name__ == '__main__':
    test_generic_get_params()
    pass
