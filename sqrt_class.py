import numpy as np
from numpy.linalg import norm

from skglm.utils import make_correlated_data
from skglm.experimental.sqrt_lasso import SqrtLasso


n_samples, n_features = 100, 200
rho = 0.01

X, y, _ = make_correlated_data(n_samples, n_features, random_state=0)

alpha_max = norm(X.T @ y, ord=np.inf) / (np.sqrt(n_samples) * norm(y))
alpha = rho * alpha_max

clf = SqrtLasso(alpha=alpha_max, verbose=2)

clf.fit(X, y)
