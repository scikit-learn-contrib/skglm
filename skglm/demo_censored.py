from skglm.datafits import Quadratic, CensoredQuadratic
from skglm.penalties import L1
from skglm.solvers import AndersonCD
from skglm.utils.jit_compilation import compiled_clone
from skglm.utils.data import make_correlated_data

X, y, _ = make_correlated_data(100, 150)

pen = compiled_clone(L1(alpha=0))

solver = AndersonCD(verbose=3)
df = Quadratic()
df = compiled_clone(df)

solver.solve(X, y, df, pen)




