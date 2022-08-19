from itertools import product
import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from skglm.utils import make_correlated_data, compiled_clone

from skglm.datafits import Logistic
from skglm.penalties import L1

from skglm.solvers.pn_solver import pn_solver
from skglm.solvers.pn_solver_improved import pn_solver_improved


@pytest.mark.parametrize('solver, X_density',
                         product([pn_solver, pn_solver_improved], [1, 0.5]))
def test_alpha_max(solver, X_density):
    n_samples, n_features = 10, 20
    X, y, _ = make_correlated_data(n_samples, n_features, X_density=X_density)
    y = np.sign(y)

    alpha_max = np.linalg.norm(X.T @ y, ord=np.inf) / (2 * n_samples)

    log_datafit = compiled_clone(Logistic())
    l1_penalty = compiled_clone(L1(alpha_max))
    w = solver(X, y, log_datafit, l1_penalty)[0]

    np.testing.assert_equal(w, 0)


@pytest.mark.parametrize("solver, rho, X_density",
                         product([pn_solver, pn_solver_improved],
                                 [1e-1, 1e-2], [1., 0.5]))
def test_pn_vs_sklearn(solver, rho, X_density):
    n_samples, n_features = 10, 20

    X, y, _ = make_correlated_data(n_samples, n_features, random_state=0,
                                   X_density=X_density)
    y = np.sign(y)

    alpha_max = np.linalg.norm(X.T @ y, ord=np.inf) / (2 * n_samples)
    alpha = rho * alpha_max

    sk_log_reg = LogisticRegression(penalty='l1', C=1/(n_samples * alpha),
                                    fit_intercept=False, tol=1e-9, solver='liblinear')
    sk_log_reg.fit(X, y)

    log_datafit = compiled_clone(Logistic())
    l1_penalty = compiled_clone(L1(alpha))
    w = solver(X, y, log_datafit, l1_penalty, tol=1e-9)[0]

    np.testing.assert_allclose(w, sk_log_reg.coef_.flatten(), rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
    pass
