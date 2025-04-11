import numpy as np
import pytest

from sklearn.covariance import GraphicalLasso as GraphicalLasso_sklearn

from skglm.covariance import GraphicalLasso, AdaptiveGraphicalLasso
from skglm.utils.data import make_dummy_covariance_data


def test_glasso_equivalence_sklearn():
    S, _, lmbd_max = make_dummy_covariance_data(200, 50)
    alpha = lmbd_max / 5

    model_sk = GraphicalLasso_sklearn(
        alpha=alpha, covariance="precomputed", tol=1e-10)
    model_sk.fit(S)

    for algo in ("primal", "dual"):
        model = GraphicalLasso(
            alpha=alpha,
            warm_start=False,
            max_iter=1000,
            tol=1e-14,
            algo=algo,
        ).fit(S)

    np.testing.assert_allclose(
        model.precision_, model_sk.precision_, atol=1e-4)
    np.testing.assert_allclose(
        model.covariance_, model_sk.covariance_, atol=1e-4)

    # check that we did not mess up lambda:
    np.testing.assert_array_less(S.shape[0] + 1, (model.precision_ != 0).sum())


def test_glasso_warm_start():
    S, _, lmbd_max = make_dummy_covariance_data(200, 50)

    alpha = lmbd_max / 5

    model = GraphicalLasso(
        alpha=alpha,
        warm_start=True,
        max_iter=1000,
        tol=1e-14,
        algo="primal",
    ).fit(S)
    np.testing.assert_array_less(1, model.n_iter_)

    model.fit(S)
    np.testing.assert_equal(model.n_iter_, 1)

    model.algo = "dual"
    with pytest.raises(ValueError, match="does not support"):
        model.fit(S)


def test_glasso_weights():
    S, _, lmbd_max = make_dummy_covariance_data(200, 50)

    alpha = lmbd_max / 10

    model = GraphicalLasso(
        alpha=alpha,
        warm_start=False,
        max_iter=2000,
        tol=1e-14,
        algo="primal",
    ).fit(S)
    prec = model.precision_.copy()

    scal = 2.
    model.weights = np.full(S.shape, scal)
    model.alpha /= scal
    model.fit(S)
    np.testing.assert_allclose(prec, model.precision_)

    mask = np.random.randn(*S.shape) > 0
    mask = mask + mask.T
    mask.flat[::S.shape[0] + 1] = 0
    weights = mask.astype(float)
    model.weights = weights
    model.fit(S)
    np.testing.assert_array_less(1e-4, np.abs(model.precision_[~mask]))


def test_glasso_adaptive():
    S, _, lmbd_max = make_dummy_covariance_data(200, 50)

    alpha = lmbd_max / 10
    tol = 1e-14
    model = GraphicalLasso(
        alpha=alpha,
        warm_start=True,
        max_iter=1000,
        tol=tol,
        algo="primal",
    ).fit(S)
    n_iter = [model.n_iter_]
    Theta1 = model.precision_
    # TODO test the other strategies
    weights = 1 / (np.abs(Theta1) + 1e-10)
    model.weights = weights

    model.fit(S)
    n_iter.append(model.n_iter_)
    print("ada:")

    # TODO test more than 2 reweightings?
    model_a = AdaptiveGraphicalLasso(
        alpha=alpha,
        n_reweights=2,
        tol=tol).fit(S)

    np.testing.assert_allclose(model_a.precision_, model.precision_)
    np.testing.assert_allclose(model_a.n_iter_, n_iter)

    # support is decreasing:
    assert not np.any(model_a.precision_[Theta1 == 0])
