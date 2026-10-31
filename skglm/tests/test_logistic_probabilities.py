import numpy as np
import pytest
from scipy.sparse import csc_matrix
from scipy.special import expit
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression

from skglm import SparseLogisticRegression


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("labels", [np.array([-3, 7]), np.array(["zebra", "ant"])])
def test_binary_predict_proba(sparse, fit_intercept, labels):
    X = np.array([[-3., -1.], [-2., 1.], [-1., 0.],
                  [1., 1.], [2., -1.], [4., 2.]])
    y = labels[[0, 0, 0, 1, 1, 1]]
    if sparse:
        X = csc_matrix(X)
    model = SparseLogisticRegression(alpha=0.1, tol=1e-10, max_iter=1000)
    model.set_params(fit_intercept=fit_intercept)
    model.fit(X, y)
    reference = LogisticRegression(
        penalty="l1", C=1 / (len(y) * model.alpha), solver="saga",
        fit_intercept=fit_intercept, tol=1e-14, max_iter=100000,
        random_state=0).fit(X.toarray() if sparse else X, y)
    np.testing.assert_allclose(model.coef_, reference.coef_, atol=1e-7)
    np.testing.assert_allclose(model.intercept_, reference.intercept_, atol=1e-7)
    probability = model.predict_proba(X)
    np.testing.assert_allclose(probability[:, 1], expit(model.decision_function(X)))
    np.testing.assert_allclose(probability, reference.predict_proba(X), atol=1e-7)
    np.testing.assert_allclose(probability.sum(axis=1), 1.)
    np.testing.assert_array_equal(model.classes_, reference.classes_)
    np.testing.assert_array_equal(
        model.classes_[probability.argmax(axis=1)], model.predict(X))


@pytest.mark.parametrize("sparse", [False, True])
def test_binary_predict_proba_zero_and_extreme_scores(sparse):
    X = np.array([[-2.], [-1.], [1.], [2.]])
    y = np.array([0, 0, 1, 1])
    model = SparseLogisticRegression(alpha=0.1, fit_intercept=False).fit(X, y)
    X_test = np.array([[-1000.], [0.], [1000.]])
    if sparse:
        X_test = csc_matrix(X_test)
    np.testing.assert_allclose(
        model.predict_proba(X_test), [[1., 0.], [0.5, 0.5], [0., 1.]])


@pytest.mark.parametrize("sparse", [False, True])
def test_multiclass_predict_proba_normalization(sparse):
    X = np.array([[-2., -1.], [-1., -2.], [2., -1.],
                  [1., -2.], [0., 2.], [1., 3.]])
    y = np.array(["a", "a", "b", "b", "c", "c"])
    if sparse:
        X = csc_matrix(X)
    model = SparseLogisticRegression(
        alpha=0.1, fit_intercept=False, tol=1e-10).fit(X, y)
    expected = expit(model.decision_function(X))
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(model.predict_proba(X), expected)
    np.testing.assert_allclose(model.predict_proba(X).sum(axis=1), 1.)


def test_predict_proba_requires_fit():
    with pytest.raises(NotFittedError):
        SparseLogisticRegression().predict_proba([[0.]])
