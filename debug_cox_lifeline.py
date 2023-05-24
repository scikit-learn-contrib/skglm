import pandas as pd

import numpy as np
from numpy.linalg import norm

from skglm.datafits import Cox
from skglm.penalties import L1
from skglm.solvers import ProxNewton

from skglm.utils.data import make_dummy_survival_data
from skglm.utils.jit_compilation import compiled_clone

from lifelines import CoxPHFitter


# generate data
reg = 1e-2
n_samples, n_features = 100, 30

tm, s, X = make_dummy_survival_data(n_samples, n_features)


# compute alpha_max
B = (tm >= tm[:, None]).astype(X.dtype)
grad_0 = -s + B.T @ (s / np.sum(B, axis=1))
alpha_max = norm(X.T @ grad_0, ord=np.inf) / n_samples

alpha = reg * alpha_max

# fit ProxNewton
datafit = compiled_clone(Cox())
penalty = compiled_clone(L1(alpha))

datafit.initialize(X, (tm, s))

w, _, _ = ProxNewton(fit_intercept=False, tol=1e-9, max_iter=100).solve(
    X, (tm, s), datafit, penalty
)

# fit lifeline estimator
df = pd.DataFrame(
    np.hstack((tm[:, None], s[:, None], X))
)

estimator = CoxPHFitter(penalizer=alpha, l1_ratio=1.)
estimator.fit(
    df,
    duration_col=0,
    event_col=1,
    show_progress=True,
    fit_options={
        "max_steps": 100_000, "precision": 1e-12
    },
)
w_ll = estimator.params_.values


# print("diff sol:", norm(w - w_ll))

p_obj_skglm = datafit.value((tm, s), w, X @ w) + penalty.value(w)
p_obj_ll = datafit.value((tm, s), w_ll, X @ w_ll) + penalty.value(w_ll)


print("norm w_ll", norm(w_ll))  # is zero, should not be because reg < 1

print("ours", p_obj_skglm)
print("them", p_obj_ll)
print("diff p_obj:", p_obj_skglm - p_obj_ll)
