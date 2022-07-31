import numpy as np
from skglm.prototype_PN.log_datafit import Pr_LogisticRegression
from skglm.utils import make_correlated_data


def test_datafit():
    n_samples, n_features = 10, 20

    w = np.ones(n_features)
    X, y, _ = make_correlated_data(n_samples, n_features)
    y = np.sign(y)
    Xw = X @ w

    log_datafit = Pr_LogisticRegression()

    grad = log_datafit.raw_gradient(y, Xw)
    hess = log_datafit.raw_hessian(y, Xw)

    np.testing.assert_equal(grad.shape, (n_samples,))
    np.testing.assert_equal(hess.shape, (n_samples,))

    np.testing.assert_almost_equal(-grad * (y + len(y) * grad), hess)


if __name__ == '__main__':
    test_datafit()
