"""
==========================================
Skglm support of scikit-learn GridSearchCV
==========================================
An example that uses scikit-learn``GridSearchCV`` to select 
the best ``alpha`` and ``l1_ratio`` of ElasticNet model.
"""

import numpy as np
from skglm import ElasticNet
from skglm.utils import make_correlated_data
from sklearn.model_selection import GridSearchCV


# Simulate dataset
n_samples, n_features = 10, 100
X, y, _ = make_correlated_data(n_samples, n_features, random_state=0)

# grid of parameter
alpha_max = np.linalg.norm(X.T @ y, ord=np.inf) / n_samples
parameters = {
    'alpha': alpha_max * np.geomspace(1, 1e-3, 100),
    'l1_ratio': [1., 0.9, 0.8, 0.7]
}

# init and fit GridSearchCV
reg = GridSearchCV(
    ElasticNet(),
    param_grid=parameters,
    cv=5, n_jobs=-1
)
reg.fit(X, y)

# print the best parameters
print(reg.best_estimator_)
