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
estimator = CoxPHFitter(penalizer=alpha, l1_ratio=1.)
estimator.fit(
    df,
    duration_col=0,
    event_col=1
)
w_ll = estimator.params_.values

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
# Now that we checked that both ``skglm`` and ``lifelines`` yield the same result,
# let's compare their execution time
import timeit

time_skglm = timeit.timeit(
    lambda: solver.solve(X, (tm, s), datafit, penalty),
    number=10
)
time_lifeline = timeit.timeit(
    lambda: estimator.fit(df, duration_col=0, event_col=1),
    number=10
)

# plot results
fig, ax = plt.subplots()

ax.bar(
    x=["skglm", "lifelines"],
    height=[time_skglm, time_lifeline],
)

# set layout of bar plot
ax.set_yscale('log')
ax.set_ylabel("time in seconds")
ax.set_title("Timing comparison")

print(f"speed up ratio {time_lifeline / time_skglm:.0f}")

# %%
# Et voilà, that is more than x100 less time to get the same solution!
