from itertools import product
from collections.abc import Iterable

import pytest
import numpy as np
import pandas as pd
import scipy.optimize
from numpy.linalg import norm
from scipy.sparse import csc_matrix, issparse
from celer import GroupLasso as GroupLasso_celer

from sklearn.base import clone
from sklearn.linear_model import Lasso as Lasso_sklearn
from sklearn.linear_model import ElasticNet as ElasticNet_sklearn
from sklearn.linear_model import LogisticRegression as LogReg_sklearn
from sklearn.linear_model import MultiTaskLasso as MultiTaskLasso_sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC as LinearSVC_sklearn
from sklearn.utils.estimator_checks import check_estimator

from skglm.utils.data import (make_correlated_data, make_dummy_survival_data,
                              _alpha_max_group_lasso, grp_converter)
from skglm.estimators import (
    GeneralizedLinearEstimator, Lasso, MultiTaskLasso, WeightedLasso, ElasticNet,
    MCPRegression, SparseLogisticRegression, LinearSVC, GroupLasso, CoxEstimator)
from skglm.datafits import Logistic, Quadratic, QuadraticSVC, QuadraticMultiTask, Cox
from skglm.penalties import L1, IndicatorBox, L1_plus_L2, MCPenalty, WeightedL1, SLOPE
from skglm.solvers import AndersonCD, FISTA, ProxNewton

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
groups = [20, 30, 10]

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

dict_estimators_sk["wMCP"] = Lasso_sklearn(
    alpha=alpha, tol=tol)
dict_estimators_ours["wMCP"] = MCPRegression(
    alpha=alpha, gamma=np.inf, tol=tol, weights=np.ones(n_features))

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
    ["Lasso", "wLasso", "ElasticNet", "MCP", "wMCP", "LogisticRegression", "SVC"])
def test_check_estimator(estimator_name):
    if estimator_name == "SVC":
        pytest.xfail("SVC check_estimator is too slow due to bug.")
    elif estimator_name == "LogisticRegression":
        # TODO: remove xfail when ProxNewton supports intercept fitting
        pytest.xfail("ProxNewton does not yet support intercept fitting")
    clf = clone(dict_estimators_ours[estimator_name])
    clf.tol = 1e-6  # failure in float32 computation otherwise
    if isinstance(clf, (WeightedLasso, MCPRegression)):
        clf.weights = None
    check_estimator(clf)


@pytest.mark.parametrize("estimator_name", dict_estimators_ours.keys())
@pytest.mark.parametrize('X', [X, X_sparse])
@pytest.mark.parametrize('fit_intercept', [True, False])
@pytest.mark.parametrize('positive', [True, False])
def test_estimator(estimator_name, X, fit_intercept, positive):
    if estimator_name == "GeneralizedLinearEstimator":
        pytest.skip()
    if fit_intercept and estimator_name == "LogisticRegression":
        pytest.xfail("sklearn LogisticRegression does not support intercept.")
    if fit_intercept and estimator_name == "SVC":
        pytest.xfail("Intercept is not supported for SVC.")
    if positive and estimator_name not in (
            "Lasso", "ElasticNet", "wLasso", "MCP", "wMCP", "GroupLasso"):
        pytest.xfail("`positive` option is only supported by L1, L1_plus_L2 and wL1.")

    estimator_sk = clone(dict_estimators_sk[estimator_name])
    estimator_ours = clone(dict_estimators_ours[estimator_name])

    estimator_sk.set_params(fit_intercept=fit_intercept)
    estimator_ours.set_params(fit_intercept=fit_intercept)

    if positive:
        estimator_sk.set_params(positive=positive)
        estimator_ours.set_params(positive=positive)

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


@pytest.mark.parametrize("use_efron, use_float_32",
                         #  product([True, False], [True, False]))
                         product([True, False], [False]))
def test_CoxEstimator(use_efron, use_float_32):
    # TODO: fix test for float_32, same for CoxEstimator_sparse
    try:
        from lifelines import CoxPHFitter
    except ModuleNotFoundError:
        pytest.xfail(
            "Testing Cox Estimator requires `lifelines` packages\n"
            "Run `pip install lifelines`"
        )

    reg = 1e-2
    # norms of solutions differ when n_features > n_samples
    n_samples, n_features = 50, 15
    random_state = 1265

    X, y = make_dummy_survival_data(n_samples, n_features, normalize=True,
                                    with_ties=use_efron, use_float_32=use_float_32,
                                    random_state=random_state)
    tm, s = y[:, 0], y[:, 1]

    # compute alpha_max
    B = (tm >= tm[:, None]).astype(X.dtype)
    grad_0 = -s + B.T @ (s / np.sum(B, axis=1))
    alpha_max = norm(X.T @ grad_0, ord=np.inf) / n_samples

    alpha = reg * alpha_max

    # fit Cox using ProxNewton solver
    datafit = Cox(use_efron)
    penalty = L1(alpha)

    # XXX: intialize is needed here although it is done in ProxNewton
    # it is used to evaluate the objective
    datafit.initialize(X, y)

    w, *_ = ProxNewton(
        fit_intercept=False, tol=1e-6, max_iter=50
    ).solve(
        X, y, datafit, penalty
    )

    # fit lifeline estimator
    df = pd.DataFrame(np.hstack((y, X)))

    estimator = CoxPHFitter(penalizer=alpha, l1_ratio=1.)
    estimator.fit(
        df, duration_col=0, event_col=1,
        fit_options={"max_steps": 10_000, "precision": 1e-12}
    )
    w_ll = estimator.params_.values.astype(X.dtype)

    p_obj_skglm = datafit.value(y, w, X @ w) + penalty.value(w)
    p_obj_ll = datafit.value(y, w_ll, X @ w_ll) + penalty.value(w_ll)

    # though norm of solution might differ
    np.testing.assert_allclose(p_obj_skglm, p_obj_ll, atol=1e-6)


@pytest.mark.parametrize("use_efron, use_float_32",
                         #  product([True, False], [True, False]))
                         product([True, False], [True]))
def test_CoxEstimator_sparse(use_efron, use_float_32):
    reg = 1e-2
    n_samples, n_features = 50, 15
    X_density, random_state = 0.5, 1265

    X, y = make_dummy_survival_data(n_samples, n_features, X_density=X_density,
                                    use_float_32=use_float_32, with_ties=use_efron,
                                    random_state=random_state)
    tm, s = y[:, 0], y[:, 1]

    # compute alpha_max
    B = (tm >= tm[:, None]).astype(X.dtype)
    grad_0 = -s + B.T @ (s / np.sum(B, axis=1))
    alpha_max = norm(X.T @ grad_0, ord=np.inf) / n_samples

    alpha = reg * alpha_max

    # fit Cox using ProxNewton solver
    datafit = Cox(use_efron)
    penalty = L1(alpha)

    *_, stop_crit = ProxNewton(
        fit_intercept=False, tol=1e-6, max_iter=50
    ).solve(
        X, y, datafit, penalty
    )

    np.testing.assert_allclose(stop_crit, 0., atol=1e-6)


@pytest.mark.parametrize("use_efron, l1_ratio", product([True, False], [1., 0.7, 0.]))
def test_Cox_sk_like_estimator(use_efron, l1_ratio):
    try:
        from lifelines import CoxPHFitter
    except ModuleNotFoundError:
        pytest.xfail(
            "Testing Cox Estimator requires `lifelines` packages\n"
            "Run `pip install lifelines`"
        )

    alpha = 1e-2
    # norms of solutions differ when n_features > n_samples
    n_samples, n_features = 100, 30
    method = "efron" if use_efron else "breslow"

    X, y = make_dummy_survival_data(n_samples, n_features, normalize=True,
                                    with_ties=use_efron, random_state=0)

    estimator_sk = CoxEstimator(
        alpha, l1_ratio=l1_ratio, method=method, tol=1e-6
    ).fit(X, y)
    w_sk = estimator_sk.coef_

    # fit lifeline estimator
    df = pd.DataFrame(np.hstack((y, X)))

    estimator_ll = CoxPHFitter(penalizer=alpha, l1_ratio=l1_ratio)
    estimator_ll.fit(
        df, duration_col=0, event_col=1,
        fit_options={"max_steps": 10_000, "precision": 1e-12}
    )
    w_ll = estimator_ll.params_.values

    # define datafit and penalty to check objs
    datafit = Cox(use_efron)
    penalty = L1_plus_L2(alpha, l1_ratio)
    datafit.initialize(X, y)

    p_obj_skglm = datafit.value(y, w_sk, X @ w_sk) + penalty.value(w_sk)
    p_obj_ll = datafit.value(y, w_ll, X @ w_ll) + penalty.value(w_ll)

    # though norm of solution might differ
    np.testing.assert_allclose(p_obj_skglm, p_obj_ll, atol=1e-6)


@pytest.mark.parametrize("use_efron, l1_ratio", product([True, False], [1., 0.7, 0.]))
def test_Cox_sk_like_estimator_sparse(use_efron, l1_ratio):
    alpha = 1e-2
    n_samples, n_features = 100, 30
    method = "efron" if use_efron else "breslow"

    X, y = make_dummy_survival_data(n_samples, n_features, X_density=0.1,
                                    with_ties=use_efron, random_state=0)

    estimator_sk = CoxEstimator(
        alpha, l1_ratio=l1_ratio, method=method, tol=1e-9
    ).fit(X, y)
    stop_crit = estimator_sk.stop_crit_

    np.testing.assert_array_less(stop_crit, 1e-8)


def test_Cox_sk_compatibility():
    check_estimator(CoxEstimator())


@pytest.mark.parametrize("use_efron, issparse", product([True, False], repeat=2))
def test_equivalence_cox_SLOPE_cox_L1(use_efron, issparse):
    # this only tests the case of SLOPE equivalent to L1 (equal alphas)
    reg = 1e-2
    n_samples, n_features = 100, 10
    X_density = 1. if not issparse else 0.2

    X, y = make_dummy_survival_data(
        n_samples, n_features, with_ties=use_efron, X_density=X_density,
        random_state=0)

    # init datafit
    datafit = Cox(use_efron)

    if not issparse:
        datafit.initialize(X, y)
    else:
        datafit.initialize_sparse(X.data, X.indptr, X.indices, y)

    # compute alpha_max
    grad_0 = datafit.raw_grad(y, np.zeros(n_samples))
    alpha_max = np.linalg.norm(X.T @ grad_0, ord=np.inf)

    # init penalty
    alpha = reg * alpha_max
    alphas = alpha * np.ones(n_features)
    penalty = SLOPE(alphas)

    solver = FISTA(opt_strategy="fixpoint", max_iter=10_000, tol=1e-9)

    w, *_ = solver.solve(X, y, datafit, penalty)

    method = 'efron' if use_efron else 'breslow'
    estimator = CoxEstimator(alpha, l1_ratio=1., method=method, tol=1e-9).fit(X, y)

    np.testing.assert_allclose(w, estimator.coef_, atol=1e-5)


@pytest.mark.parametrize("use_efron", [True, False])
def test_cox_SLOPE(use_efron):
    reg = 1e-2
    n_samples, n_features = 100, 10

    X, y = make_dummy_survival_data(
        n_samples, n_features, with_ties=use_efron, random_state=0)

    # init datafit
    datafit = Cox(use_efron)
    datafit.initialize(X, y)

    # compute alpha_max
    grad_0 = datafit.raw_grad(y, np.zeros(n_samples))
    alpha_ref = np.linalg.norm(X.T @ grad_0, ord=np.inf)

    # init penalty
    alpha = reg * alpha_ref
    alphas = alpha / np.arange(n_features + 1)[1:]
    penalty = SLOPE(alphas)

    solver = FISTA(opt_strategy="fixpoint", max_iter=10_000, tol=1e-9)

    w, *_ = solver.solve(X, y, datafit, penalty)

    result = scipy.optimize.minimize(
        fun=lambda w: datafit.value(y, w, X @ w) + penalty.value(w),
        x0=np.zeros(n_features),
        method="SLSQP",
        options=dict(
            ftol=1e-9,
            maxiter=10_000,
        ),
    )
    w_sp = result.x

    # check both methods yield the same objective
    np.testing.assert_allclose(
        datafit.value(y, w, X @ w) + penalty.value(w),
        datafit.value(y, w_sp, X @ w_sp) + penalty.value(w_sp)
    )


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
        solver = AndersonCD(tol=tol, fit_intercept=fit_intercept)
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
        Datafit(), Penalty(1.), AndersonCD(fit_intercept=False)).fit(X, y)
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


@pytest.mark.parametrize("fit_intercept, issparse",
                         product([False, True], [False, True]))
def test_GroupLasso_estimator(fit_intercept, issparse):
    reg = 1e-1
    grp_indices, grp_ptr = grp_converter(groups, X.shape[1])
    n_groups = len(grp_ptr) - 1
    weights = np.abs(np.random.randn(n_groups))
    alpha = reg * _alpha_max_group_lasso(X, y, grp_indices, grp_ptr, weights)

    estimator_ours = GroupLasso(groups=groups, alpha=alpha, tol=tol,
                                weights=weights, fit_intercept=fit_intercept)
    estimator_celer = GroupLasso_celer(groups=groups, alpha=alpha, tol=tol,
                                       weights=weights, fit_intercept=fit_intercept)

    X_ = csc_matrix(X) if issparse else X

    estimator_celer.fit(X_, y)
    estimator_ours.fit(X_, y)
    coef_celer = estimator_celer.coef_
    coef_ours = estimator_ours.coef_

    np.testing.assert_allclose(coef_ours, coef_celer, atol=1e-4, rtol=1e-2)
    np.testing.assert_allclose(estimator_celer.intercept_,
                               estimator_ours.intercept_, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("fit_intercept, issparse",
                         product([False, True], [True, False]))
def test_GroupLasso_estimator_positive(fit_intercept, issparse):
    reg = 1e-1
    grp_indices, grp_ptr = grp_converter(groups, X.shape[1])
    n_groups = len(grp_ptr) - 1
    weights = np.abs(np.random.randn(n_groups))
    alpha = reg * _alpha_max_group_lasso(X, y, grp_indices, grp_ptr, weights)

    estimator_ours = GroupLasso(groups=groups, alpha=alpha, tol=tol,
                                weights=weights, fit_intercept=fit_intercept,
                                positive=True)

    X_ = csc_matrix(X) if issparse else X
    estimator_ours.fit(X_, y)

    # check all coefs are positive
    coef_ = estimator_ours.coef_
    np.testing.assert_equal(len(coef_[coef_ < 0]), 0)
    # check optimality
    np.testing.assert_array_less(estimator_ours.stop_crit_, tol)


@pytest.mark.parametrize("positive", [False, True])
def test_GroupLasso_estimator_sparse_vs_dense(positive):
    reg = 1e-1
    grp_indices, grp_ptr = grp_converter(groups, X.shape[1])
    n_groups = len(grp_ptr) - 1
    weights = np.abs(np.random.randn(n_groups))
    alpha = reg * _alpha_max_group_lasso(X, y, grp_indices, grp_ptr, weights)

    glasso = GroupLasso(groups=groups, alpha=alpha, tol=1e-8,
                        weights=weights, positive=positive)

    glasso.fit(X, y)
    coef_dense = glasso.coef_

    X_sparse = csc_matrix(X)
    glasso.fit(X_sparse, y)
    coef_sparse = glasso.coef_

    np.testing.assert_allclose(coef_sparse, coef_dense, atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize("X, l1_ratio", product([X, X_sparse], [1., 0.7, 0.]))
def test_SparseLogReg_elasticnet(X, l1_ratio):

    estimator_sk = clone(dict_estimators_sk['LogisticRegression'])
    estimator_ours = clone(dict_estimators_ours['LogisticRegression'])
    estimator_sk.set_params(fit_intercept=True, solver='saga',
                            penalty='elasticnet', l1_ratio=l1_ratio, max_iter=10_000)
    estimator_ours.set_params(fit_intercept=True, l1_ratio=l1_ratio, max_iter=10_000)

    estimator_sk.fit(X, y)
    estimator_ours.fit(X, y)
    coef_sk = estimator_sk.coef_
    coef_ours = estimator_ours.coef_

    np.testing.assert_array_less(1e-5, norm(coef_ours))
    np.testing.assert_allclose(coef_ours, coef_sk, atol=1e-6)
    np.testing.assert_allclose(
        estimator_sk.intercept_, estimator_ours.intercept_, rtol=1e-4)


if __name__ == "__main__":
    pass
