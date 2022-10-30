import numpy as np
from numpy.linalg import norm
from skglm.utils import make_correlated_data
from skglm.estimators import MultiTaskLasso


n_samples = 3
n_features = 4
n_tasks = 5
X, Y, _ = make_correlated_data(
        n_samples=n_samples, n_features=n_features, n_tasks=n_tasks, 
        density=0.5, random_state=0)
alpha_max = np.max(norm(X.T @ Y, axis=1, ord=2)) / n_samples
alpha = alpha_max * 0.1

clf = MultiTaskLasso(
        alpha, positive=True, fit_intercept=False, tol=1e-8, 
        ws_strategy="subdiff", verbose=2).fit(X, Y)

print(clf.coef_.T)

