from itertools import product
from collections.abc import Iterable

import pytest
import numpy as np
from numpy.linalg import norm

from sklearn.base import clone
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
from skglm.solvers import AcceleratedCD


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
    alpha=alpha, tol=tol)
dict_estimators_ours["Lasso"] = Lasso(
    alpha=alpha, tol=tol)

dict_estimators_sk["wLasso"] = Lasso_sklearn(
    alpha=alpha, tol=tol)
dict_estimators_ours["wLasso"] = WeightedLasso(
    alpha=alpha, tol=tol, weights=np.ones(n_features))

dict_estimators_sk["ElasticNet"] = ElasticNet_sklearn(
    alpha=alpha, l1_ratio=l1_ratio, tol=tol)
dict_estimators_ours["ElasticNet"] = ElasticNet(
    alpha=alpha, l1_ratio=l1_ratio, tol=tol)

dict_estimators_sk["MCP"] = Lasso_sklearn(
    alpha=alpha, tol=tol)
dict_estimators_ours["MCP"] = MCPRegression(
    alpha=alpha, gamma=np.inf, tol=tol)

dict_estimators_sk["LogisticRegression"] = LogReg_sklearn(
    C=1/(alpha * n_samples), tol=tol, penalty='l1',
    solver='liblinear')
dict_estimators_ours["LogisticRegression"] = SparseLogisticRegression(
    alpha=alpha, tol=tol)

C = 1.
dict_estimators_sk["SVC"] = LinearSVC_sklearn(
    penalty='l2', loss='hinge', fit_intercept=False, dual=True, C=C, tol=tol)
dict_estimators_ours["SVC"] = LinearSVC(C=C, tol=tol)


@pytest.mark.parametrize(
    "estimator_name",
    ["Lasso", "wLasso", "ElasticNet", "MCP", "LogisticRegression", "SVC"])
def test_check_estimator(estimator_name):
    if estimator_name == "SVC":
        pytest.xfail("SVC check_estimator is too slow due to bug.")
    elif estimator_name == "LogisticRegression":
        # TODO: remove xfail when ProxNewton supports intercept fitting
        pytest.xfail("ProxNewton does not yet support intercept fitting")
    clf = clone(dict_estimators_ours[estimator_name])
    clf.tol = 1e-6  # failure in float32 computation otherwise
    if isinstance(clf, WeightedLasso):
        clf.weights = None
    check_estimator(clf)


@pytest.mark.parametrize("estimator_name", dict_estimators_ours.keys())
@pytest.mark.parametrize('X', [X, X_sparse])
@pytest.mark.parametrize('fit_intercept', [True, False])
def test_estimator(estimator_name, X, fit_intercept):
    if estimator_name == "GeneralizedLinearEstimator":
        pytest.skip()
    if fit_intercept and estimator_name == "LogisticRegression":
        pytest.xfail("sklearn LogisticRegression does not support intercept.")
    if fit_intercept and estimator_name == "SVC":
        pytest.xfail("Intercept is not supported for SVC.")

    estimator_sk = clone(dict_estimators_sk[estimator_name])
    estimator_ours = clone(dict_estimators_ours[estimator_name])

    estimator_sk.set_params(fit_intercept=fit_intercept)
    estimator_ours.set_params(fit_intercept=fit_intercept)

    estimator_sk.fit(X, y)
    estimator_ours.fit(X, y)
    coef_sk = estimator_sk.coef_
    coef_ours = estimator_ours.coef_

    np.testing.assert_array_less(1e-5, norm(coef_ours))
    np.testing.assert_allclose(coef_ours, coef_sk, atol=1e-6)
    np.testing.assert_allclose(
        estimator_sk.intercept_, estimator_ours.intercept_, rtol=1e-4)
    if fit_intercept:
        np.testing.assert_array_less(1e-4, estimator_ours.intercept_)


@pytest.mark.parametrize('X, fit_intercept', product([X, X_sparse], [True, False]))
def test_mtl_vs_sklearn(X, fit_intercept):
    estimator_sk = MultiTaskLasso_sklearn(
        alpha, fit_intercept=fit_intercept, tol=1e-8)
    estimator_ours = MultiTaskLasso(
        alpha, fit_intercept=fit_intercept, tol=1e-8)

    # sklearn does not support sparse X:
    estimator_sk.fit(X.toarray() if issparse(X) else X, Y)
    estimator_ours.fit(X, Y)
    coef_sk = estimator_sk.coef_
    coef_ours = estimator_ours.coef_
    np.testing.assert_allclose(coef_ours, coef_sk, atol=1e-4)
    np.testing.assert_allclose(
        estimator_ours.intercept_, estimator_sk.intercept_, rtol=1e-4)


# TODO also add a test for the sparse case?
def test_mtl_path():
    fit_intercept = False  # sklearn cannot fit an intercept in path. It is done
    # only in their fit method.
    alphas = np.geomspace(alpha_max, alpha_max * 0.01, 10)
    coef_sk = MultiTaskLasso_sklearn(
        fit_intercept=fit_intercept).path(
            X, Y, l1_ratio=1, tol=1e-14, max_iter=5_000, alphas=alphas
    )[1][:, :X.shape[1]]
    coef_ours = MultiTaskLasso(fit_intercept=fit_intercept, tol=1e-14).path(
        X, Y, alphas, max_iter=10)[1][:, :X.shape[1]]
    np.testing.assert_allclose(coef_ours, coef_sk, rtol=1e-5)


# Test if GeneralizedLinearEstimator returns the correct coefficients
@pytest.mark.parametrize("Datafit, Penalty, Estimator, pen_args", [
    (Quadratic, L1, Lasso, [alpha]),
    (Quadratic, WeightedL1, WeightedLasso,
     [alpha, np.random.choice(3, n_features)]),
    (Quadratic, L1_plus_L2, ElasticNet, [alpha, 0.3]),
    (Quadratic, MCPenalty, MCPRegression, [alpha, 3]),
    (QuadraticSVC, IndicatorBox, LinearSVC, [alpha]),
    (Logistic, L1, SparseLogisticRegression, [alpha]),
])
@pytest.mark.parametrize('fit_intercept', [True, False])
def test_generic_estimator(fit_intercept, Datafit, Penalty, Estimator, pen_args):
    if isinstance(Datafit(), QuadraticSVC) and fit_intercept:
        pytest.xfail()
    elif Datafit == Logistic and fit_intercept:
        pytest.xfail("TODO support intercept in Logistic datafit")
    else:
        solver = AcceleratedCD(tol=tol, fit_intercept=fit_intercept)
        target = Y if Datafit == QuadraticMultiTask else y
        gle = GeneralizedLinearEstimator(
            Datafit(), Penalty(*pen_args), solver).fit(X, target)
        est = Estimator(
            *pen_args, tol=tol, fit_intercept=fit_intercept).fit(X, target)
        np.testing.assert_allclose(gle.coef_, est.coef_, rtol=1e-5)
        np.testing.assert_allclose(gle.intercept_, est.intercept_)


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
        Datafit(), Penalty(1.), AcceleratedCD(fit_intercept=False)).fit(X, y)
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

    reg = GeneralizedLinearEstimator(Quadratic(), L1(4.))
    clf = GeneralizedLinearEstimator(Logistic(), MCPenalty(2., 3.))

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
    estimator_sk = clone(dict_estimators_sk[estimator_name])
    estimator_ours = clone(dict_estimators_ours[estimator_name])
    estimator_sk.tol = 1e-10
    # XXX: No need for `tol` anymore as it already is in solver
    estimator_ours.tol = 1e-10
    estimator_sk.max_iter = 10_000
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


@pytest.mark.parametrize(
    "estimator_name",
    ["Lasso", "wLasso", "ElasticNet", "MCP", "LogisticRegression", "SVC"])
def test_warm_start(estimator_name):
    if estimator_name == "LogisticRegression":
        # TODO: remove xfail when ProxNewton supports intercept fitting
        pytest.xfail("ProxNewton does not yet support intercept fitting")
    model = clone(dict_estimators_ours[estimator_name])
    model.warm_start = True
    model.fit(X, y)
    np.testing.assert_array_less(0, model.n_iter_)
    model.fit(X, y)  # already fitted + warm_start so 0 iter done
    np.testing.assert_equal(0, model.n_iter_)


if __name__ == "__main__":
    pass
