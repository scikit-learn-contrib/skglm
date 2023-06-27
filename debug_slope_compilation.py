import numpy as np
np.set_printoptions(precision=3)

from skglm.penalties import SLOPE
from skglm.utils.jit_compilation import compiled_clone

np.random.seed(0)
d = 10
alphas = np.ones(d)

x = np.random.randn(d)

pen = SLOPE(alphas=alphas[::-1])

# print(res.x)
skg = pen.prox_vec(x, 1)
print(skg)
comp = compiled_clone(pen)
print(comp.prox_vec(x, 1))


