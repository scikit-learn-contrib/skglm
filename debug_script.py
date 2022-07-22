import numpy as np
from skglm.datafits import Quadratic, Quadratic_32
from skglm.utils import make_correlated_data


X, y, _ = make_correlated_data()

X = X.astype(np.float32)
y = y.astype(np.float32)

print(X.dtype, y.dtype)

# This will throw an error
# one should use Quadratic_32 instead
quad = Quadratic()
quad.initialize(X, y)
