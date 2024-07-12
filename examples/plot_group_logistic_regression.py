"""
=================================
Fast Sparse Group Lasso in python
=================================
Scikit-learn is missing a Sparse Group Lasso regression estimator. We show how to
implement one with ``skglm``.
"""

# Author: Mathurin Massias

# %%
import numpy as np
import matplotlib.pyplot as plt

from skglm.solvers import GroupBCD
from skglm.datafits import QuadraticGroup
from skglm import GeneralizedLinearEstimator
from skglm.penalties import WeightedSparseGroupL2
from skglm.utils.data import make_correlated_data, grp_converter

n_features = 30
X, y, _ = make_correlated_data(
    n_samples=10, n_features=30, random_state=0)


# %%
# Model creation: combination of penalty, datafit and solver.
#
# penalty:
grp_size = 10  # take groups of 10 consecutive features
n_groups = n_features // grp_size
grp_indices, grp_ptr = grp_converter(grp_size, n_features)
n_groups = len(grp_ptr) - 1
weights_g = np.ones(n_groups, dtype=np.float64)
weights_f = 0.5 * np.ones(n_features)
penalty = WeightedSparseGroupL2(
    alpha=0.5, weights_groups=weights_g,
    weights_features=weights_f, grp_indices=grp_indices, grp_ptr=grp_ptr)

# %% Datafit and solver
datafit = QuadraticGroup(grp_ptr, grp_indices)
solver = GroupBCD(ws_strategy="fixpoint", verbose=1, fit_intercept=False, tol=1e-10)

model = GeneralizedLinearEstimator(datafit, penalty, solver=solver)

# %%
# Train the model
clf = GeneralizedLinearEstimator(datafit, penalty, solver)
clf.fit(X, y)

# %%
# Some groups are fully 0, and inside non zero groups,
# some values are 0 too
plt.imshow(clf.coef_.reshape(-1, grp_size) != 0, cmap='Greys')
plt.title("Non zero values (in black) in model coefficients")
plt.ylabel('Group index')
plt.xlabel('Feature index inside group')
plt.xticks(np.arange(grp_size))
plt.yticks(np.arange(n_groups));

# %%
