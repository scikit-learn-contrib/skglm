import numpy as np
from time import time
from skglm import GeneralizedLinearEstimator
from skglm import datafits
from skglm import penalties
from skglm.solvers import ProxNewton

rng = np.random.default_rng()

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
popu_X = simu_X(n, p, example='hete')
# MM: if this is not added, the solver gives nans
# popu_X /= 2

popu_Y = simu_Y(popu_X, n, beta=beta_value, type='poisson')
# popu_Y += 1
print('simulate data time:', time() - start_time)

solver = ProxNewton(verbose=3, max_iter=50, tol=1e-6)
start_time = time()


alpha_max = (popu_X.T @ (1 - popu_Y) / len(popu_Y)).max()
print("alpha max: %1e" % alpha_max)
# alpha = 1.1 * alpha_max  # This yields only 0 coefficients
alpha = alpha_max / 100

model = GeneralizedLinearEstimator(
    datafit=datafits.Poisson(), penalty=penalties.L1(alpha=alpha),
    solver=solver).fit(X=popu_X, y=popu_Y)
result = model.coef_
print('fit time time:', time() - start_time)
print( np.where(result !=0)[0] )
print( result[np.where(result !=0)[0]] )
