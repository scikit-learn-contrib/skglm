import numpy as np
from time import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from numpy.linalg import norm

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


n = 500
p = 2000
beta_value = np.array([5, 3, 0, 0, -2, 0] + [0] * (p - 6))

start_time = time()
X = simu_X(n, p, example='hete')
X /= 10

y = simu_Y(X, n, beta=beta_value, type='poisson')
print('simulate data time:', time() - start_time)

datafit = datafits.Poisson()
penalty = penalties.L1(alpha=1)
alpha_max = penalty.alpha_max(datafit.gradient(X, y, np.zeros(len(y))))
print(f'{alpha_max=}')

penalty.alpha = alpha_max / 100

solver = ProxNewton(verbose=1, max_iter=50, warm_start=True, fit_intercept=False,
                    tol=1e-4)

###### single fit:
start_time = time()

model = GeneralizedLinearEstimator(
    datafit=datafit, penalty=penalty, solver=solver).fit(X, y)
result = model.coef_
print(f"fit time : {(time() - start_time):.2e} s")
print( np.where(result !=0)[0] )
print( result[np.where(result !=0)[0]] )


###### Grid Search
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
alpha_grid = np.geomspace(alpha_max, alpha_max / 10000, num=200)

rmses_train = []
rmses_test = []
loss_train = []
loss_test = []
for alpha in alpha_grid:
    penalty.alpha = alpha
    model.fit(X_train, y_train)
    rmses_test.append(norm(y_test - model.predict(X_test)) / np.sqrt(X.shape[0]))
    rmses_train.append(norm(y_train - model.predict(X_train)) / np.sqrt(X.shape[0]))
    loss_train.append(datafit.value(y_train, model.coef_, X_train @ model.coef_))
    loss_test.append(datafit.value(y_test, model.coef_, X_test @ model.coef_))

plt.close()
plt.loglog(alpha_grid, rmses_test, label='test RMSE', color="tab10:orange")
plt.loglog(alpha_grid, rmses_train, label='train RMSE', color="tab10:orange", linestyle='--')
plt.loglog(alpha_grid, loss_test, label='test datafitting loss', color="tab10:blue")
plt.loglog(alpha_grid, loss_train, label='train datafitting loss',
           color="tab10:blue", linestyle='--')
plt.legend()
plt.xlabel("regularization strength alpha")
plt.show(block=False)