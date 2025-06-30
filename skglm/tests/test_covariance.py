import numpy as np
import pytest

from sklearn.covariance import GraphicalLasso as GraphicalLasso_sklearn

from skglm.covariance import GraphicalLasso, AdaptiveGraphicalLasso
from skglm.utils.data import make_dummy_covariance_data
from skglm.penalties.separable import LogSumPenalty


def test_glasso_equivalence_sklearn():
    S, X, Theta_true, lmbd_max = make_dummy_covariance_data(200, 50)
    alpha = lmbd_max / 5

    model_sk = GraphicalLasso_sklearn(
        alpha=alpha, tol=1e-10)
    model_sk.fit(X)

    for algo in ("primal", "dual"):
        model = GraphicalLasso(
            alpha=alpha,
            warm_start=False,
            max_iter=1000,
            tol=1e-14,
            algo=algo,
        ).fit(X)

        np.testing.assert_allclose(
            model.precision_, model_sk.precision_, atol=1e-4)
        np.testing.assert_allclose(
            model.covariance_, model_sk.covariance_, atol=1e-4)

    # check that we did not mess up lambda:
    np.testing.assert_array_less(X.shape[1] + 1, (model.precision_ != 0).sum())


def test_glasso_warm_start():
    S, X, Theta_true, lmbd_max = make_dummy_covariance_data(200, 50)

    alpha = lmbd_max / 5

    model = GraphicalLasso(
        alpha=alpha,
        warm_start=True,
        max_iter=1000,
        tol=1e-14,
        algo="primal",
    ).fit(X)
    np.testing.assert_array_less(1, model.n_iter_)

    model.fit(X)
    np.testing.assert_equal(model.n_iter_, 1)

    model.algo = "dual"
    with pytest.raises(ValueError, match="does not support"):
        model.fit(X)


def test_glasso_weights():
    S, X, Theta_true, lmbd_max = make_dummy_covariance_data(200, 50)

    alpha = lmbd_max / 10

    model = GraphicalLasso(
        alpha=alpha,
        warm_start=False,
        max_iter=2000,
        tol=1e-16,
        inner_tol=1e-10,
        algo="primal",
    ).fit(S, mode='precomputed')
    prec = model.precision_.copy()

    scal = 2.
    model.weights = np.full(S.shape, scal)
    model.alpha /= scal
    model.fit(S, mode='precomputed')
    np.testing.assert_allclose(prec, model.precision_)

    mask = np.random.randn(*S.shape) > 0
    mask = mask + mask.T
    mask.flat[::S.shape[0] + 1] = 0
    weights = mask.astype(float)
    model.weights = weights
    model.fit(S, mode='precomputed')
    np.testing.assert_array_less(1e-10, np.abs(model.precision_[~mask]))


def test_glasso_adaptive():
    S, X, Theta_true, lmbd_max = make_dummy_covariance_data(200, 50)

    alpha = lmbd_max / 10
    tol = 1e-14
    eps = 1e-10
    model = GraphicalLasso(
        alpha=alpha,
        warm_start=True,
        max_iter=1000,
        tol=tol,
        algo="primal",
    ).fit(X)
    n_iter = [model.n_iter_]
    Theta1 = model.precision_
    # TODO test the other strategies
    weights = 1 / (np.abs(Theta1) + eps)
    model.weights = weights

    model.fit(X)
    n_iter.append(model.n_iter_)
    print("ada:")

    # TODO test more than 2 reweightings?
    model_a = AdaptiveGraphicalLasso(
        alpha=alpha,
        penalty=LogSumPenalty(alpha=alpha, eps=eps),
        n_reweights=2,
        tol=tol).fit(X)

    np.testing.assert_allclose(model_a.precision_, model.precision_, rtol=1e-10)

    # support is decreasing:
    assert not np.any(model_a.precision_[Theta1 == 0])
