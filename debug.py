import numpy as np
from time import time
from skglm import GeneralizedLinearEstimator
from skglm import datafits
from skglm import penalties
from skglm.solvers import ProxNewton

# MM: make sure to seed
rng = np.random.default_rng(0)

def simu_X(n, p, example):
    if example == 'iid':
       return rng.normal(size=(n, p))
    elif example == 'hete':
        mean_vec = np.zeros(p)
        cov_mat = 0.5 ** np.abs( np.arange(1, p + 1).reshape(-1,1) - np.arange(1, p + 1) )
        return rng.multivariate_normal(mean_vec, cov_mat, n, method='cholesky')

def simu_Y(X, n, beta, type):
    match type:
        case 'linear':
            return X @ beta + rng.normal(size=n)
        case 'logistic':
            probe = 1 / (1 + np.exp(-X @ beta))
            return rng.binomial(n=1, p=probe, size=n)
        case 'poisson':
            probe = np.exp(X @ beta)
            return rng.poisson(lam=probe, size=n)


n = 200
p = 5000
beta_value = np.array([5, 3, 0, 0, -2, 0] + [0] * (p - 6))

start_time = time()
X = simu_X(n, p, example='hete')

y = simu_Y(X, n, beta=beta_value, type='poisson')
print('simulate data time:', time() - start_time)

datafit = datafits.Poisson()
penalty = penalties.L1(alpha=1)
alpha_max = penalty.alpha_max(datafit.gradient(X, y, np.zeros(len(y))))
print(f'{alpha_max=}')

penalty.alpha = alpha_max / 100

solver = ProxNewton(verbose=1, max_iter=50, fit_intercept=False)
start_time = time()


# alpha_max = (X.T @ (1 - y) / len(y)).max()
# print("alpha max: %1e" % alpha_max)
# alpha = 1.1 * alpha_max  # This yields only 0 coefficients
alpha = alpha_max / 100

model = GeneralizedLinearEstimator(
    datafit=datafit, penalty=penalty, solver=solver).fit(X, y)
result = model.coef_
print(f"fit time : {(time() - start_time):.2e} s")
print( np.where(result !=0)[0] )
print( result[np.where(result !=0)[0]] )
