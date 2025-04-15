import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression

from skglm.penalties import L1
from skglm.datafits import Logistic
from skglm.solvers.prox_newton import ProxNewton

from skglm.utils.data import make_correlated_data


@pytest.mark.parametrize("X_density", [1., 0.5])
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("ws_strategy", ["subdiff", "fixpoint"])
def test_pn_vs_sklearn(X_density, fit_intercept, ws_strategy):
    n_samples, n_features = 12, 25
    rho = 1e-1

    X, y, _ = make_correlated_data(n_samples, n_features, random_state=0,
                                   X_density=X_density)
    y = np.sign(y)

    alpha_max = np.linalg.norm(X.T @ y, ord=np.inf) / (2 * n_samples)
    alpha = rho * alpha_max

    sk_log_reg = LogisticRegression(penalty='l1', C=1/(n_samples * alpha),
                                    fit_intercept=fit_intercept, random_state=0,
                                    tol=1e-12, solver='saga', max_iter=1_000_000)
    sk_log_reg.fit(X, y)

    log_datafit = Logistic()
    l1_penalty = L1(alpha)
    prox_solver = ProxNewton(
        fit_intercept=fit_intercept, tol=1e-12, ws_strategy=ws_strategy)
    w = prox_solver.solve(X, y, log_datafit, l1_penalty)[0]

    np.testing.assert_allclose(w[:n_features], sk_log_reg.coef_.flatten())
    if fit_intercept:
        np.testing.assert_allclose(w[-1], sk_log_reg.intercept_)


if __name__ == '__main__':
    pass
