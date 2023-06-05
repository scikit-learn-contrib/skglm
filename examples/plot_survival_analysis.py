# Authors: Badr Moufad
#          Mathurin Massias
"""
========================================================
Comparison of lifelines with skglm for survival analysis
========================================================
This example shows that skglm find the same solution as lifelines in 100 x less time.
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
axes[0].set_ylabel("density")