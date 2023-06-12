import numpy as np

from skglm.solvers import LBFGS
from skglm.penalties import L2
from skglm.datafits import Logistic

from sklearn.linear_model import LogisticRegression

from skglm.utils.data import make_correlated_data
from skglm.utils.jit_compilation import compiled_clone


def test_lbfgs_L2_logreg():
    reg = 1.
    n_samples, n_features = 50, 10

    X, y, _ = make_correlated_data(
        n_samples, n_features, random_state=0)
    y = np.sign(y)

    datafit = compiled_clone(Logistic())
    penalty = compiled_clone(L2(reg))
    solver = LBFGS(verbose=1)

    w, *_ = solver.solve(X, y, datafit, penalty)

    # fit scikit learn
    estimator = LogisticRegression(
        penalty='l2',
        C=1 / (n_samples * reg),
        fit_intercept=False
    )
    estimator.fit(X, y)

    np.testing.assert_allclose(
        w, estimator.coef_.flatten(), atol=1e-4
    )


if __name__ == "__main__":
    test_lbfgs_L2_logreg()
    pass
