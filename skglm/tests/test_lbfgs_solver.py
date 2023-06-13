import pytest
import numpy as np
import pandas as pd

from skglm.solvers import LBFGS
from skglm.penalties import L2
from skglm.datafits import Logistic, Cox

from sklearn.linear_model import LogisticRegression

from skglm.utils.jit_compilation import compiled_clone
from skglm.utils.data import make_correlated_data, make_dummy_survival_data


def test_lbfgs_L2_logreg():
    reg = 1.
    n_samples, n_features = 50, 10

    X, y, _ = make_correlated_data(
        n_samples, n_features, random_state=0)
    y = np.sign(y)

    # fit L-BFGS
    datafit = compiled_clone(Logistic())
    penalty = compiled_clone(L2(reg))
    w, *_ = LBFGS().solve(X, y, datafit, penalty)

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


@pytest.mark.parametrize("use_efron", [True, False])
def test_L2_Cox(use_efron):
    try:
        from lifelines import CoxPHFitter
    except ModuleNotFoundError:
        pytest.xfail(
            "Testing Cox Estimator requires `lifelines` packages\n"
            "Run `pip install lifelines`"
        )

    alpha = 1.
    n_samples, n_features = 50, 10
    random_state = 1265

    tm, s, X = make_dummy_survival_data(n_samples, n_features,
                                        with_ties=use_efron,
                                        random_state=random_state)

    datafit = compiled_clone(Cox(use_efron))
    penalty = compiled_clone(L2(alpha))

    datafit.initialize(X, (tm, s))
    w, *_ = LBFGS().solve(X, (tm, s), datafit, penalty)

    # fit lifeline estimator
    stacked_tm_s_X = np.hstack((tm[:, None], s[:, None], X))
    df = pd.DataFrame(stacked_tm_s_X)

    estimator = CoxPHFitter(penalizer=alpha, l1_ratio=0.)
    estimator.fit(
        df, duration_col=0, event_col=1,
        fit_options={"max_steps": 10_000, "precision": 1e-20},
        show_progress=True,
    )
    w_ll = estimator.params_.values

    p_obj_skglm = datafit.value((tm, s), w, X @ w) + penalty.value(w)
    p_obj_ll = datafit.value((tm, s), w_ll, X @ w_ll) + penalty.value(w_ll)

    # though norm of solution might differ
    np.testing.assert_allclose(p_obj_skglm, p_obj_ll, atol=1e-3)


if __name__ == "__main__":
    pass
