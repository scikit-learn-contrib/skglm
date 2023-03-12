import numpy as np
from skglm.gpu.utils.host_utils import compute_obj


def test_compute_obj():

    # generate dummy data
    X = np.eye(3)
    y = np.array([1, 0, 1])
    w = np.array([1, 2, -3])
    lmbd = 10.

    p_obj = compute_obj(X, y, lmbd, w)

    np.testing.assert_array_equal(p_obj, 0.5 * 20 + 10. * 6)
