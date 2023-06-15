import numpy as np
from numpy.linalg import norm

from skglm.datafits import Cox

from skglm.utils.data import make_dummy_survival_data
from skglm.utils.jit_compilation import compiled_clone


use_efron = True
reg = 1e-2
n_samples, n_features = 100, 10
random_state = 1265
rng = np.random.RandomState(random_state)

tm, s, X = make_dummy_survival_data(n_samples, n_features, normalize=True,
                                    with_ties=use_efron, random_state=random_state)

# build explicitly the matrix B
B = (tm >= tm[:, None]).astype(X.dtype)

# init datafit
datafit = compiled_clone(Cox(use_efron))
datafit.initialize(X, (tm, s))

# check correctness for several `vec`
for _ in range(10):
    print("================")
    print("================")

    vec = rng.randn(n_samples)
    print(
        "check B @ v:", norm(datafit._B_dot_vec(vec) - B @ vec)
    )

    print(
        "check B.T @ vec:", norm(datafit._B_T_dot_vec(vec) - B.T @ vec)
    )
