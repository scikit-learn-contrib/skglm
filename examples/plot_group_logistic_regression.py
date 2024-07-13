"""
===================================
Group Logistic regression in python
===================================
Scikit-learn is missing a Group Logistic regression estimator. We show how to implement
one with ``skglm``.
"""

# Author: Mathurin Massias

import numpy as np

from skglm import GeneralizedLinearEstimator
from skglm.datafits import LogisticGroup
from skglm.penalties import WeightedGroupL2
from skglm.solvers import GroupProxNewton
from skglm.utils.data import make_correlated_data, grp_converter

n_features = 30
X, y, _ = make_correlated_data(
    n_samples=10, n_features=30, random_state=0)
y = np.sign(y)


# %%
# Classifier creation: combination of penalty, datafit and solver.
#
grp_size = 3  # groups are made of groups of 3 consecutive features
n_groups = n_features // grp_size
grp_indices, grp_ptr = grp_converter(grp_size, n_features=n_features)
alpha = 0.01
weights = np.ones(n_groups)
penalty = WeightedGroupL2(alpha, weights, grp_ptr, grp_indices)
datafit = LogisticGroup(grp_ptr, grp_indices)
solver = GroupProxNewton(verbose=2)

# %%
# Train the model
clf = GeneralizedLinearEstimator(datafit, penalty, solver)
clf.fit(X, y)

# %%
# Fit check that groups are either all 0 or all non zero
print(clf.coef_.reshape(-1, grp_size))
