import numpy as np
from numpy.linalg import norm

from skglm.utils import make_correlated_data
from skglm.experimental.sqrt_lasso import SqrtLasso


n_samples, n_features = 100, 200
# rho = 0.15  # this gives high enough residual norm
rho = 0.1  # solver gets stuck for this one

X, y, _ = make_correlated_data(n_samples, n_features, random_state=0)

alpha_max = norm(X.T @ y, ord=np.inf) / (np.sqrt(n_samples) * norm(y))
alpha = rho * alpha_max

clf = SqrtLasso(alpha=alpha, verbose=2)

clf.fit(X, y)

# print(clf.coef_)
print(norm(y - clf.predict(X)))
