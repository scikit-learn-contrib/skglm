import pytest
import numpy as np
import pandas as pd

from skglm.penalties import L2
from skglm.solvers import LBFGS
from skglm.datafits import Logistic, Cox

from sklearn.linear_model import LogisticRegression

from skglm.utils.jit_compilation import compiled_clone
from skglm.utils.data import make_correlated_data, make_dummy_survival_data


@pytest.mark.parametrize("X_sparse", [True, False])
def test_lbfgs_L2_logreg(X_sparse):
    reg = 1.
    X_density = 1. if not X_sparse else 0.5
    n_samples, n_features = 100, 50

    X, y, _ = make_correlated_data(
        n_samples, n_features, random_state=0, X_density=X_density,
    )
    y = np.sign(y)

    # fit L-BFGS
    datafit = compiled_clone(Logistic())
    penalty = compiled_clone(L2(reg))
    w, *_ = LBFGS(tol=1e-12).solve(X, y, datafit, penalty)

    # fit scikit learn
    estimator = LogisticRegression(
        penalty='l2',
        C=1 / (n_samples * reg),
        fit_intercept=False,
        tol=1e-12,
    ).fit(X, y)

    np.testing.assert_allclose(w, estimator.coef_.flatten())


@pytest.mark.parametrize("use_efron", [True, False])
def test_L2_Cox(use_efron):
    try:
        from lifelines import CoxPHFitter
    except ModuleNotFoundError:
        pytest.xfail(
            "Testing L2 Cox Estimator requires `lifelines` packages\n"
            "Run `pip install lifelines`"
        )

    alpha = 10.
    n_samples, n_features = 100, 50

    tm, s, X = make_dummy_survival_data(
        n_samples, n_features, normalize=True,
        with_ties=use_efron, random_state=0)

    datafit = compiled_clone(Cox(use_efron))
    penalty = compiled_clone(L2(alpha))

    datafit.initialize(X, (tm, s))
    w, *_ = LBFGS().solve(X, (tm, s), datafit, penalty)

    # fit lifeline estimator
    stacked_tm_s_X = np.hstack((tm[:, None], s[:, None], X))
    df = pd.DataFrame(stacked_tm_s_X)

    estimator = CoxPHFitter(penalizer=alpha, l1_ratio=0.).fit(
        df, duration_col=0, event_col=1
    )
    w_ll = estimator.params_.values

    p_obj_skglm = datafit.value((tm, s), w, X @ w) + penalty.value(w)
    p_obj_ll = datafit.value((tm, s), w_ll, X @ w_ll) + penalty.value(w_ll)

    # despite increasing tol in lifelines, solutions are quite far apart
    # suspecting lifelines https://github.com/CamDavidsonPilon/lifelines/pull/1534
    # as our solution gives the lowest objective value
    np.testing.assert_allclose(w, w_ll, rtol=1e-1)
    np.testing.assert_allclose(p_obj_skglm, p_obj_ll, rtol=1e-6)


if __name__ == "__main__":
    pass
