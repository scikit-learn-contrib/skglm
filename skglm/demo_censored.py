import numpy as np


from skglm.datafits import Quadratic, CensoredQuadratic
from skglm.penalties import L1
from skglm.solvers import AndersonCD
from skglm.utils.jit_compilation import compiled_clone
from skglm.utils.data import make_correlated_data

X, y, _ = make_correlated_data(100, 150)

pen = compiled_clone(L1(alpha=0))

solver = AndersonCD(verbose=3, fit_intercept=True)
df = Quadratic()
df = compiled_clone(df)

w = solver.solve(X, y, df, pen)[0]

df2 = CensoredQuadratic(X.T @ y, y.mean())
df2 = compiled_clone(df2)

w2 = solver.solve(X, np.zeros(X.shape[0]), df2, pen)[0]
np.testing.assert_allclose(w2, w)