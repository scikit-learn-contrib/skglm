import numpy as np
from numpy.linalg import norm
from skglm.penalties import L1
from skglm.datafits.single_task import SqrtQuadratic, Logistic
from skglm.utils import make_correlated_data, compiled_clone

from skglm.solvers.prox_newton import prox_newton

from statsmodels.regression import linear_model
from skglm.utils import make_correlated_data
from numpy.linalg import norm


n_samples, n_features = 1000, 200
rho = 0.1

X, y, _ = make_correlated_data(n_samples, n_features, random_state=0, snr=3)

alpha_max = norm(X.T @ y, ord=np.inf) / (np.sqrt(n_samples) * norm(y))
alpha = rho * alpha_max

sqrt_quad = compiled_clone(SqrtQuadratic())
l1_penalty = compiled_clone(L1(alpha=alpha))

w = prox_newton(X, y, sqrt_quad, l1_penalty, tol=1e-9, max_epochs=20, verbose=0)[0]


# model = linear_model.OLS(y, X)
# model = model.fit_regularized(method='sqrt_lasso', L1_wt=1., alpha=n_samples * alpha)

# w_stats = [4.21552919e-03, -3.90760035e-08,  2.84013553e-01,  8.37413377e-09,
#            3.93440141e-10,  6.34859422e-01,  1.28859848e-09, -7.42144006e-02,
#            8.85829556e-02,  6.50662716e-02,  1.58602049e-08,  9.38828525e-01,
#            1.08050786e-08,  1.67343023e-01, -2.98894287e-03, -8.91879344e-08,
#            5.71381684e-09, -5.99768494e-10,  2.54943430e-01, -1.02551038e-01]

# print(norm(w - w_stats, ord=np.inf))
