import numpy as np
from skglm.gpu.utils.host_utils import compute_obj, eval_opt_crit
from skglm.gpu.utils.device_utils import get_device_properties

from sklearn.linear_model import Lasso


def test_compute_obj():

    # generate dummy data
    X = np.eye(3)
    y = np.array([1, 0, 1])
    w = np.array([1, 2, -3])
    lmbd = 10.

    p_obj = compute_obj(X, y, lmbd, w)

    np.testing.assert_array_equal(p_obj, 20 / (2 * 3) + 10. * 6)


def test_eval_optimality():
    rng = np.random.RandomState(1235)
    n_samples, n_features = 10, 5

    X = rng.randn(n_samples, n_features)
    y = rng.randn(n_samples)
    lmbd = 1.

    estimator = Lasso(
        alpha=lmbd, fit_intercept=False, tol=1e-9
    ).fit(X, y)

    np.testing.assert_allclose(
        eval_opt_crit(X, y, lmbd, estimator.coef_), 0.,
        atol=1e-9
    )


def test_device_props():
    # check it runs and result is a dict
    dev_props = get_device_properties()
    np.testing.assert_equal(type(dev_props), dict)
