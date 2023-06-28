import numpy as np
from skglm.penalties import SLOPE

from skglm.utils.jit_compilation import compiled_clone
from skglm.utils.prox_funcs import ST_vec
np.set_printoptions(precision=3)


# params
alphas = np.array([1, 1, 1], dtype=np.float64)
x = np.array([1.7, 0.4, 1.9], dtype=np.float64)
pen = SLOPE(alphas=alphas)

print(
    f"{'x':20}", x
)

skg = pen.prox_vec(x, 1)
print(
    f"{'SLOPE not compiled':20}", skg
)

comp = compiled_clone(pen)
print(
    f"{'SLOPE compiled':20}", comp.prox_vec(x, 1))

print(
    f"{'ST':20}", ST_vec(x, 1.)
)
