# %%
import numpy as np
from skglm.datafits import HessianQuadratic, Quadratic
from skglm.solvers import AndersonCD
from skglm.penalties import L1
from skglm.estimators import L1PenalizedQP, Lasso
import scipy as sp
# %%
n = 100
d = 3
G = np.random.randn(n, d)
A = G.T @ G + np.eye(d)
b = np.random.randn(d)
U, s, Vh = sp.linalg.svd(A, full_matrices=False)
S = np.diag(s)
A_half = U @ np.diag(s**(0.5)) @ Vh
A_minus_half = U @ np.diag(s**(-0.5)) @ Vh
X = np.sqrt(d)*A_half
y = -np.sqrt(d)*A_minus_half @ b
# %%
x = np.random.randn(d)
Ax = A @ x
hessian_quadratic_loss = HessianQuadratic()
hessian_quadratic_loss.value(b, x, Ax)
# %%
Xw = X @ x
quadratic_loss = Quadratic()
quadratic_loss.value(y, None, Xw) - (1.0/(2*d))*(y**2).sum()

# %%
alpha = 1e-1
qp_solver = L1PenalizedQP(alpha=alpha, fit_intercept=False, max_iter=500)
lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=500)

qp_solver.fit(A, b)
lasso.fit(X, y)
# %%
print(f'{qp_solver.coef_=}')
print(f'{lasso.coef_=}')
