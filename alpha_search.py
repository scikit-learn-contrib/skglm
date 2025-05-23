import numpy as np
from numpy.linalg import norm
from skglm import Lasso
from skglm.experimental import SqrtLasso

np.random.seed(0)
X = np.random.randn(10, 20)
y = np.random.randn(10)
y += 1

n = len(y)

alpha_max = norm(X.T @ y, ord=np.inf) / n

alpha = alpha_max / 10

lass = Lasso(alpha=alpha, fit_intercept=True, tol=1e-8).fit(X, y)
w_lass = lass.coef_
assert norm(w_lass) > 0

scal = n / norm(y - lass.predict(X))

sqrt = SqrtLasso(alpha=alpha * scal, fit_intercept=True, tol=1e-8).fit(X, y)

print(norm(w_lass - sqrt.coef_))


# diffs = []
# alphas = np.linspace(16.07, 16.08, num=50)
# for scal in alphas:
#     sqrt = SqrtLasso(alpha=alpha * scal).fit(X, y)
#     diffs.append(norm(w_lass - sqrt.coef_))

# best = np.argmin(diffs)
# print(alphas[best])
# print(diffs[best])
