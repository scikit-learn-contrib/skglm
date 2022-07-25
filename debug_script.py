import numpy as np
from skglm.datafits import Quadratic
from skglm.utils import make_correlated_data

from numba import int32, float64
from skglm.datafits.base import spec_to_float32


# X, y, _ = make_correlated_data()

# X = X.astype(np.float32)
# y = y.astype(np.float32)

# print(X.dtype, y.dtype)

# # This will throw an error
# # one should use Quadratic_32 instead
# quad = Quadratic()
# quad.initialize(X, y)


spec_QuadraticGroup = [
    ('grp_ptr', int32[:]),
    ('grp_indices', int32[:]),
    ('lipschitz', float64[:])
]

# int are converted to float
print(spec_to_float32(spec_QuadraticGroup))
