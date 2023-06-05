# Authors: Badr Moufad
#          Mathurin Massias
"""
========================================================
Comparison of lifelines with skglm for survival analysis
========================================================
This example shows that ``skglm`` find the same solution as ``lifelines``
in x100 less time.
"""

# %%
# Data
# ----
#
# Let's first generate synthetic data using ``skglm`` data utils.
# Let's start with generating synthetic data on which to run Cox estimator.
# ``skglm`` exposes functions to generate dummy dataset among which
# ``make_dummy_survival_data`` to generate data for survival analysis problems.
#
from skglm.utils.data import make_dummy_survival_data

n_samples, n_features = 1000, 100
tm, s, X = make_dummy_survival_data(
    n_samples, n_features,
    normalize=True,
    random_state=1235
)

# %%
# The synthetic data has the following properties:
#
# * ``tm`` is the vector occurrences times which follows a Weibull(1) distribution
# * ``s`` indicates the observations censorship and follows a Bernoulli(0.5)
# * ``X`` the matrix of predictors generated using standard normal distribution
#
# Let's check this out through a histogram
import matplotlib.pyplot as plt

# init figure
fig, axes = plt.subplots(
    1, 3,
    tight_layout=True
)

# plot histograms
dists = (tm, s, X[:, 5])
axes_title = ("times", "censorship", "fifth predictor")

for idx, (dist, name) in enumerate(zip(dists, axes_title)):
    axes[idx].hist(dist, bins="auto")
    axes[idx].set_title(name)

# format y axis
axes[0].set_ylabel("count")

# %%
# Fit Cox Estimator
# -----------------
#
# After generating the synthetic data, we can now fit a L1-Cox estimator.
# Todo so, we need to combine a Cox datafit and and :math:`\ell_1` penalty
# and solve the resulting problem using Proximal Newton solver ``ProxNewton``.
# We set the intensity of the :math:`\ell_1` regularization to ``alpha=1e-3``.
from skglm.datafits import Cox
from skglm.penalties import L1
from skglm.solvers import ProxNewton

from skglm.utils.jit_compilation import compiled_clone

# set intensity of regularization
alpha = 1e-3

# init datafit and penalty
datafit = compiled_clone(Cox())
penalty = compiled_clone(L1(alpha))

datafit.initialize(X, (tm, s))

# init solver
solver = ProxNewton(
    fit_intercept=False,
    max_iter=50,
)

# solve the problem
w_sk, *_ = solver.solve(
    X, (tm, s),
    datafit,
    penalty
)

# %%
# Let's do the same with ``lifelines`` through its ``CoxPHFitter``
# estimator and compare the objectives.
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

# format data
stacked_tm_s_X = np.hstack((tm[:, None], s[:, None], X))
df = pd.DataFrame(stacked_tm_s_X)

# fit lifelines estimator
lifelines_estimator = CoxPHFitter(penalizer=alpha, l1_ratio=1.)
lifelines_estimator.fit(
    df,
    duration_col=0,
    event_col=1
)
w_ll = lifelines_estimator.params_.values

# %%
# Ideally the values of the objectives should the same, in other terms the difference
# must be close to zero.
obj_sk = datafit.value((tm, s), w_sk, X @ w_sk) + penalty.value(w_sk)
obj_ll = datafit.value((tm, s), w_ll, X @ w_ll) + penalty.value(w_ll)

print(f"objective skglm: {obj_sk:.6f}")
print(f"objective lifelines: {obj_ll:.6f}")
print(f"Difference: {abs(obj_sk - obj_ll):.6f}")
# %%
# We can do the same to check how close the two solutions are.
print(f"Difference solutions: {np.linalg.norm(w_sk - w_ll):.3e}")

# %%
# Timing comparison
# -----------------
#
# Now that we checked that both ``skglm`` and ``lifelines`` yield the same results,
# let's compare their execution time.
import time
import warnings

# ignore warnings
warnings.filterwarnings('ignore')

# where to save records
records = {
    "skglm": {
        "times": [], "objs": []
    },
    "lifelines": {
        "times": [], "objs": []
    },
}

# time skglm
max_runs = 20
for n_iter in range(1, max_runs + 1):
    solver.max_iter = n_iter

    start = time.perf_counter()
    w, *_ = solver.solve(
        X, (tm, s),
        datafit,
        penalty
    )
    end = time.perf_counter()

    records["skglm"]["objs"].append(
        datafit.value((tm, s), w, X @ w) + penalty.value(w)
    )
    records["skglm"]["times"].append(end - start)

# time lifelines
max_runs = 50
for n_iter in range(1, max_runs + 1):
    solver.max_iter = n_iter

    start = time.perf_counter()
    lifelines_estimator.fit(
        df,
        duration_col=0,
        event_col=1,
        fit_options={"max_steps": n_iter}
    )
    end = time.perf_counter()

    w = lifelines_estimator.params_.values

    records["lifelines"]["objs"].append(
        datafit.value((tm, s), w, X @ w) + penalty.value(w)
    )
    records["lifelines"]["times"].append(end - start)


# cast records as numpy array
for idx, label in enumerate(("skglm", "lifelines")):
    for metric in ("objs", "times"):
        records[label][metric] = np.asarray(records[label][metric])

# %%
# Plot the results

# init figure
fig, axes = plt.subplots(
    2, 1,
    sharex=True,
    tight_layout=True,
)

labels = ("skglm", "lifelines")
colors = ("#1f77b4", "#d62728")

optimal_obj = min(records[label]["objs"].min() for label in labels)

# plot evolution of suboptimality
for label, color in zip(labels, colors):
    axes[0].plot(
        records[label]["times"],
        records[label]["objs"] - optimal_obj,
        label=label,
        color=color,
        marker='o',
    )

# plot total time
axes[1].barh(
    [0, 1],
    [records[label]["times"][-1] for label in labels],
    color=colors
)
axes[1].set_yticks([0, 1], labels=labels)


# set figure layout
axes[0].set_yscale('log')
axes[0].set_xscale('log')
axes[1].set_xscale('log')

axes[0].set_ylabel("suboptimality")
axes[1].set_xlabel("time in seconds")
